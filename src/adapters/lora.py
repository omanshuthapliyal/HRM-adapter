"""
LoRA (Low-Rank Adaptation) adapter -- baseline fine-tuning method.

Implements LoRALinear, a drop-in replacement for nn.Linear that adds a low-rank residual:
  output = W_0 @ x + (alpha/rank) * B @ A @ x
  - A initialized N(0, 1/rank), B initialized to zero -> zero adapter output at init
  - Only A and B are trainable; W_0 is frozen

Public API:
  LoRALinear(in_features, out_features, rank, alpha, base_layer=None)
    -- wraps an existing nn.Linear (frozen) or creates a new one
  layer.merge()     -- absorb B@A into W_0 for zero-overhead inference (in-place)
  layer.unmerge()   -- reverse merge (in-place)
  layer.forward(x)  -- standard forward pass
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a low-rank residual adapter.

    W_eff = W_0 + (alpha/rank) * lora_B @ lora_A

    W_0 is always frozen. lora_A and lora_B are the only trainable parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        base_layer: nn.Linear | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank
        self.merged = False

        # Frozen base weight -- either copied from an existing layer or freshly initialised
        if base_layer is not None:
            # Share the weight tensor so in-place merge modifies the original storage
            self.weight = base_layer.weight
            self.bias = base_layer.bias
        else:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features), requires_grad=False
            )
            nn.init.normal_(self.weight, std=0.02)
            self.bias = None

        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # LoRA matrices
        # lora_A: N(0, 1/rank) -- small but non-zero so gradients flow from step 0
        # lora_B: zeros         -- adapter output is exactly 0 at initialisation
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) / math.sqrt(rank)
        )
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        if self.merged:
            # LoRA contribution already baked into self.weight -- don't double-add
            return base_out
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + self.scale * lora_out

    def merge(self) -> None:
        """Absorb lora_B @ lora_A into W_0 for zero-overhead inference."""
        if self.merged:
            return
        self.weight.data += self.scale * (self.lora_B @ self.lora_A)
        self.merged = True

    def unmerge(self) -> None:
        """Reverse a previous merge."""
        if not self.merged:
            return
        self.weight.data -= self.scale * (self.lora_B @ self.lora_A)
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.3f}, merged={self.merged}"
        )
