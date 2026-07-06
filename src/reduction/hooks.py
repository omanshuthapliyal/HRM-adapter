"""
Forward hook utilities for SSM hidden state collection (calibration phase).

Registers PyTorch forward hooks on HRMAdapter modules to capture hidden state
trajectories h_0..h_T during a calibration forward pass over the calibration split.
States are collected by intercepting the SSM's per-step computation via a wrapper.

Collected states feed into EmpiricalGrammians (grammians.py) and DMD (dmd.py).

Public API:
  StateCollector(model)
    .register()        -- attach hooks to all HRMAdapter instances in model
    .remove()          -- detach hooks (call after calibration is done)
    .get_snapshots()   -- dict[layer_name -> Tensor(N*T, state_dim)]
    .clear()           -- reset collected buffers
"""

import torch
import torch.nn as nn
from typing import Dict, List


class StateCollector:
    """
    Collects SSM hidden state trajectories from all HRMAdapter modules in a model.

    Usage:
        collector = StateCollector(model)
        collector.register()
        with torch.no_grad():
            for x, _, _ in calib_loader:
                model(x.to(device))
        collector.remove()
        snapshots = collector.get_snapshots()  # {name: Tensor(N*T, state_dim)}
    """

    def __init__(self, model: nn.Module):
        self.model    = model
        self._hooks:  List  = []
        self._buffers: Dict[str, List[torch.Tensor]] = {}

    # ------------------------------------------------------------------

    def register(self) -> None:
        """Attach hooks to every HRMAdapter found in the model."""
        # Import here to avoid circular imports
        from src.adapters.hrm_adapter import HRMAdapter

        self._hooks  = []
        self._buffers = {}

        for name, module in self.model.named_modules():
            if isinstance(module, HRMAdapter):
                self._buffers[name] = []
                hook = module.ssm.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """Return a forward hook that appends all per-step hidden states."""
        def hook(module, inputs, output):
            # output is the SSM's return tensor: (B, T, output_dim)
            # We need to re-run the recurrence to collect h_t.
            # Simpler: patch SSM.forward to expose states via a side channel.
            # Here we use a lightweight approach: re-run _get_discrete_matrices
            # and replay the recurrence on the same input to extract h.
            x = inputs[0]                            # (B, T, input_dim)
            B, T, _ = x.shape
            A_bar, B_bar = module._get_discrete_matrices()
            h = x.new_zeros(B, module.state_dim)
            states = []
            with torch.no_grad():
                for t in range(T):
                    u = x[:, t, :]
                    h = h * A_bar + u @ B_bar.T      # (B, state_dim)
                    states.append(h)
            # states: list of T tensors (B, state_dim) -> stack -> (B, T, state_dim)
            traj = torch.stack(states, dim=1)        # (B, T, state_dim)
            # Flatten batch and time for Grammian accumulation
            flat = traj.reshape(-1, module.state_dim)   # (B*T, state_dim)
            self._buffers[layer_name].append(flat.cpu().detach())

        return hook

    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Detach all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_snapshots(self) -> Dict[str, torch.Tensor]:
        """
        Returns {layer_name: Tensor(N*T, state_dim)} -- all collected hidden states
        concatenated across calibration batches.
        """
        return {
            name: torch.cat(chunks, dim=0)
            for name, chunks in self._buffers.items()
            if chunks
        }

    def clear(self) -> None:
        """Reset collected buffers (allows re-use after a new register() call)."""
        self._buffers = {name: [] for name in self._buffers}
