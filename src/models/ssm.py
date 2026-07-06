"""
Custom small SSM (State Space Model) -- S4D-Real / Mamba-inspired, written from scratch.

Parameterization:
  - A: diagonal, parameterized as A = -exp(log_A) to guarantee stability (eigenvalues < 0)
    Initialized via S4D-Real: log_A_n = log(n+1), giving eigenvalues -(1), -(2), ..., -(N)
    This spreads time constants from tau=1 (slow, n=0) to tau=1/N (fast, n=N-1), enabling
    the SSM to capture memory at multiple timescales simultaneously -- the core inductive
    bias for accumulation tasks.  Ref: Gu et al., "How to Train Your HiPPO" (2022); S4D.
  - B, C: input-independent linear projections (non-selective baseline for POC)
    Initialized at O(1/sqrt(dim)) scale so gradient signal is not killed at step 0.
  - Delta: learnable step size per channel, constrained via softplus
  - Discretization: Zero-Order Hold (ZOH)
      A_bar = exp(Delta * A)
      B_bar = (A_bar - 1) / A * B   [diagonal A simplification]

Intended use: standalone module, wrapped by HRMAdapter (src/adapters/hrm_adapter.py).
After Balanced Truncation, A/B/C matrices are replaced with reduced versions in-place.

Public API:
  SSM(input_dim, state_dim, output_dim, dt_init, dt_min, dt_max)
  ssm.forward(x)             -- (B, T, input_dim) -> (B, T, output_dim)
  ssm.step(x_t, h)           -- single-step: (B, D), (B, d) -> y_t, h_next
  ssm.get_matrices()         -- returns (A_bar, B_bar, C) as tensors for Grammian extraction
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSM(nn.Module):
    """
    Minimal linear SSM with ZOH discretization and S4D-Real-inspired initialization.

    Continuous-time system:
        dh/dt = A h(t) + B u(t)
        y(t)  = C h(t)

    where A = diag(a), a_i = -exp(log_A_i)  (always negative -> stable)

    S4D-Real initialization: log_A_n = log(n+1)  ->  a_n = -(n+1)
    Time constant of mode n: tau_n = 1/(n+1)*dt, ranging from slow (n=0) to fast (n=N-1).
    This multi-scale initialization is what allows the SSM to capture both short- and
    long-range dependencies and is essential for accumulation / memory tasks.

    ZOH discretization with per-channel step Delta:
        A_bar_i = exp(delta_i * a_i)            in (0, 1)
        B_bar_i = (A_bar_i - 1) / a_i * B_i    (row i of B, scaled)

    Recurrence:
        h_k = A_bar * h_{k-1} + B_bar @ u_k     (element-wise * for diagonal A_bar)
        y_k = C @ h_k
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        dt_init: float = 0.01,
        dt_min: float = 1e-4,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.dt_min = dt_min
        self.dt_max = dt_max

        # S4D-Real: log_A_n = log(n+1)  ->  a_n = -(n+1)
        # Eigenvalues spread from -1 (slow mode) to -N (fast mode).
        # Uniform log_A=0 (old default) puts all modes at the same timescale -- wrong.
        log_A_init = torch.log(torch.arange(1, state_dim + 1).float())
        self.log_A = nn.Parameter(log_A_init)

        # B: (state_dim, input_dim)
        # Mamba/S4D initialize continuous B at O(1) so that after ZOH discretization
        # B_bar = (A_bar-1)/a * B ~= dt * B remains O(dt) ~= O(0.01-0.1).
        # With the old 0.01 scale, B_bar ~= 1e-4 which kills the input coupling entirely.
        self.B = nn.Parameter(torch.randn(state_dim, input_dim))

        # C: (output_dim, state_dim) -- 1/sqrt(state_dim) so state->output is unit-variance
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) / math.sqrt(state_dim))

        # log_dt: log-uniform in [log(dt_min), log(dt_max)] -- unchanged, already correct
        log_dt = (
            torch.rand(state_dim) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.log_dt = nn.Parameter(log_dt)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _get_discrete_matrices(self):
        """Return (A_bar, B_bar) after ZOH discretization."""
        a = -torch.exp(self.log_A)                                   # (state_dim,) < 0
        delta = F.softplus(self.log_dt).clamp(self.dt_min, self.dt_max)  # (state_dim,)

        A_bar = torch.exp(delta * a)                                 # (state_dim,) in (0,1)
        scale = (A_bar - 1.0) / a                                    # (state_dim,) > 0
        B_bar = scale.unsqueeze(1) * self.B                          # (state_dim, input_dim)

        return A_bar, B_bar

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Parallel scan (replaces sequential for-loop)
    # ------------------------------------------------------------------

    @staticmethod
    def _fft_scan(
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes h_t = sum_{k=0}^{t} A_bar^{t-k} * (B_bar @ u_k) for all t in parallel
        using FFT-based causal convolution (O(T log T) vs O(T) sequential dispatches).

        Since A_bar is time-invariant (diagonal), the recurrence is equivalent to:
            h_t = (g * b)[t],  g_j = A_bar^j,  b_k = B_bar @ u_k
        i.e. a causal linear convolution with an exponentially-decaying filter.

        Correctness verified by: h_0 = b_0, h_1 = A_bar*b_0 + b_1 (matches sequential).
        Gradients flow through both g (via log_A) and b (via B_bar, x) automatically.
        """
        _, T, _ = x.shape
        # b_t = B_bar @ u_t for all t at once -- single batched matmul
        bs = x @ B_bar.T                                        # (B, T, state_dim)

        # Convolution filter: g_k = A_bar^k, shape (T, state_dim)
        ks = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)  # (T, 1)
        g  = torch.exp(A_bar.log().unsqueeze(0) * ks)          # (T, state_dim)

        # Zero-pad to 2T to convert circular FFT into linear (causal) convolution
        T2  = 2 * T
        G   = torch.fft.rfft(g,  n=T2, dim=0).unsqueeze(0)    # (1, T2//2+1, d)
        BS  = torch.fft.rfft(bs, n=T2, dim=1)                  # (B, T2//2+1, d)
        hs  = torch.fft.irfft(G * BS, n=T2, dim=1)             # (B, 2T, d) -- real output
        return hs[:, :T, :]                                     # (B, T, state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        Returns: (B, T, output_dim)
        State is reset to zero at the start of each call (training mode).
        Uses FFT-based parallel scan: O(T log T) vs O(T) sequential dispatches.
        """
        A_bar, B_bar = self._get_discrete_matrices()
        hs = self._fft_scan(A_bar, B_bar, x)    # (B, T, state_dim)
        return hs @ self.C.T                     # (B, T, output_dim)

    def step(self, x_t: torch.Tensor, h: torch.Tensor):
        """
        Single-step recurrence for autoregressive inference.
        x_t: (B, input_dim)
        h:   (B, state_dim)
        Returns: y_t (B, output_dim), h_next (B, state_dim)
        """
        A_bar, B_bar = self._get_discrete_matrices()
        h_next = h * A_bar + x_t @ B_bar.T         # (B, state_dim)
        y_t = h_next @ self.C.T                    # (B, output_dim)
        return y_t, h_next

    def get_matrices(self):
        """
        Return detached discretized matrices for Grammian extraction.
        Returns: (A_bar, B_bar, C)
            A_bar : (state_dim,)            diagonal entries
            B_bar : (state_dim, input_dim)
            C     : (output_dim, state_dim)
        """
        A_bar, B_bar = self._get_discrete_matrices()
        return A_bar.detach(), B_bar.detach(), self.C.detach()
