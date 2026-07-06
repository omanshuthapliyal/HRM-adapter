"""
HRMAdapter -- Hankel-Rank-Mamba adapter module.

Wraps an SSM (src/models/ssm.py) and injects it as a residual parallel to a Transformer
sub-layer (MLP by default):
  output = gate * SSM(x)
where gate is a learnable scalar initialized to 0.0 (zero-residual init -- adapter is
transparent at the start of fine-tuning, preserving the frozen base model's behaviour).

Post Balanced Truncation, replace_ssm_matrices() hot-swaps the SSM's (A_bar, B_bar, C)
with the reduced versions directly in discrete-domain, bypassing ZOH recomputation.

State management:
  - Training:  stateless per call (state reset to zero each forward)
  - Inference: rolling hidden_state stored across calls (set via reset_state())

Public API:
  HRMAdapter(ssm, input_dim, output_dim)
  adapter.forward(x)                      -- (B, T, D) -> (B, T, D) residual output
  adapter.replace_ssm_matrices(A_bar, B_bar, C)  -- hot-swap post-BT (discrete domain)
  adapter.reset_state(batch_size, device)  -- allocate rolling hidden state for inference
  adapter.clear_state()                   -- zero the rolling hidden state
"""

import math

import torch
import torch.nn as nn

from src.models.ssm import SSM


class HRMAdapter(nn.Module):
    """
    Residual SSM adapter with zero-gate initialisation.

    During training: stateless BPTT (hidden state reset each call).
    After replace_ssm_matrices(): uses reduced (A_bar, B_bar, C) directly,
    bypassing ZOH so the discretization step does not un-reduce the matrices.
    """

    def __init__(self, ssm: SSM, input_dim: int, output_dim: int,
                 gate_init: float = 0.0):
        super().__init__()
        self.ssm        = ssm
        self.input_dim  = input_dim
        self.output_dim = output_dim

        # gate_init=0 -> zero-residual (base unaffected); >0 -> non-zero gradient for SSM body from step 0
        self.gate = nn.Parameter(torch.full((1,), gate_init))

        # Post-BT reduced matrices (discrete domain) -- None until replace_ssm_matrices()
        self._reduced       = False
        self._A_bar_reduced = None   # (d_hat,)
        self._B_bar_reduced = None   # (d_hat, input_dim)
        self._C_reduced     = None   # (output_dim, d_hat)

        # Rolling hidden state for inference (None during training)
        self.hidden_state = None

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        Returns: (B, T, output_dim)  -- gated SSM residual

        Three modes:
          - hidden_state is None, not reduced: stateless FFT scan (training)
          - hidden_state is set, not reduced:  stateful sequential scan (chunked inference)
          - reduced: uses reduced (A_bar, B_bar, C) matrices
        """
        if self._reduced:
            out = self._forward_reduced(x)
        elif self.hidden_state is not None:
            out = self._forward_stateful(x)
        else:
            out = self.ssm(x)          # (B, T, output_dim)

        return (self.gate * out).to(x.dtype)

    def _forward_stateful(self, x: torch.Tensor) -> torch.Tensor:
        """Stateful sequential scan using full SSM matrices. Updates hidden_state in-place.

        Everything runs in float32: casting h to x.dtype each call then back to float32
        for storage accumulates rounding error across chunks (fp16 has ~3 decimal digits
        of precision vs float32's ~7). hidden_state is always stored as float32.
        """
        A_bar, B_bar = self.ssm._get_discrete_matrices()   # float32 params
        x32 = x.float()
        h = self.hidden_state.float()                       # always float32
        outputs = []
        for t in range(x32.shape[1]):
            h = h * A_bar + x32[:, t, :] @ B_bar.T
            outputs.append(h @ self.ssm.C.T)
        self.hidden_state = h.detach()                      # store float32
        return torch.stack(outputs, dim=1).to(x.dtype)     # (B, T, output_dim)

    def _forward_reduced(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using post-BT discrete matrices directly (no ZOH).
        A: (d_hat, d_hat) full matrix OR (d_hat,) diagonal -- both are handled.
        B_bar: (d_hat, input_dim)
        C:     (output_dim, d_hat)

        Training (hidden_state is None, diagonal A): FFT parallel scan, O(T log T).
        Inference (hidden_state set) or full-matrix A: sequential for-loop.
        Full-matrix A has d_hat<=25 post-BT so the loop is cheap.
        """
        B_sz, T, _ = x.shape
        A     = self._A_bar_reduced   # (d_hat,) or (d_hat, d_hat)
        B_bar = self._B_bar_reduced   # (d_hat, input_dim)
        C     = self._C_reduced       # (output_dim, d_hat)

        # Fast path: diagonal A, no rolling state (training mode)
        if A.dim() == 1 and self.hidden_state is None:
            hs = SSM._fft_scan(A, B_bar, x)    # (B, T, d_hat)
            return hs @ C.T                    # (B, T, output_dim)

        # Slow path: full-matrix A or inference with rolling state
        d_hat = A.shape[0]
        h = self.hidden_state if self.hidden_state is not None else x.new_zeros(B_sz, d_hat)
        outputs = []
        for t in range(T):
            u = x[:, t, :]
            h = h * A + u @ B_bar.T if A.dim() == 1 else h @ A.T + u @ B_bar.T
            outputs.append(h @ C.T)

        if self.hidden_state is not None:
            self.hidden_state = h.detach()

        return torch.stack(outputs, dim=1)      # (B, T, output_dim)

    # ------------------------------------------------------------------

    def replace_ssm_matrices(
        self,
        A_bar: torch.Tensor,   # (d_hat,) diagonal OR (d_hat, d_hat) full matrix
        B_bar: torch.Tensor,   # (d_hat, input_dim)
        C:     torch.Tensor,   # (output_dim, d_hat)
    ) -> None:
        """
        Hot-swap to BT-reduced matrices.  Switches the adapter into reduced mode:
        future forward() calls use _forward_reduced() instead of ssm.forward().

        A_bar can be:
          - (d_hat,)         diagonal vector  -> stored as-is, used with elementwise *
          - (d_hat, d_hat)   full matrix      -> stored as-is, used with matrix multiply
        """
        device = next(self.parameters()).device
        self._A_bar_reduced = A_bar.to(device).detach()
        self._B_bar_reduced = B_bar.to(device).detach()
        self._C_reduced     = C.to(device).detach()
        self._reduced       = True

    # ------------------------------------------------------------------

    def reset_state(self, batch_size: int, device: torch.device = None) -> None:
        """Allocate / zero the rolling hidden state for autoregressive inference."""
        if device is None:
            device = next(self.parameters()).device
        if self._reduced:
            d_hat = self._A_bar_reduced.shape[0]  # works for both 1D and 2D A
        else:
            d_hat = self.ssm.state_dim
        self.hidden_state = torch.zeros(batch_size, d_hat, device=device)

    def clear_state(self) -> None:
        """Clear the rolling hidden state (switch back to stateless mode)."""
        self.hidden_state = None

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        reduced_dim = (
            self._A_bar_reduced.shape[0] if self._reduced else "full"
        )
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"state={'reduced:' + str(reduced_dim) if self._reduced else 'full:' + str(self.ssm.state_dim)}, "
            f"gate={self.gate.item():.4f}"
        )
