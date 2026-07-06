"""
Grammian computation for data-driven model order reduction.

Two modes:
  1. Exact (preferred): analytic Lyapunov solution for diagonal-A discrete LTI systems.
     Valid because our SSM guarantees A = diag(a_i) with |a_i| < 1.
       Wc_ij = (B B^T)_ij / (1 - a_i * a_j)     [discrete reachability Lyapunov]
       Wo_ij = (C^T C)_ij / (1 - a_i * a_j)     [discrete observability Lyapunov]

  2. Empirical (fallback): snapshot-based estimate useful when A is unknown.
       Wc ~ (1/NT) * H^T @ H
       Wo ~ (1/NT) * (H @ C^T)^T @ (H @ C^T)    [H @ C^T gives output signals]

Both return (state_dim, state_dim) symmetric PSD matrices suitable for bt_reduce().

Public API:
  compute_exact_grammians(A_bar, B_bar, C) -> (Wc, Wo)
    A_bar : (state_dim,)              diagonal entries of discrete A, values in (0,1)
    B_bar : (state_dim, input_dim)
    C     : (output_dim, state_dim)

  compute_grammians(snapshots, C_matrix) -> (Wc, Wo)
    snapshots : (NT, state_dim)        concatenated hidden states from calibration
    C_matrix  : (output_dim, state_dim)
"""

import torch


def compute_exact_grammians(
    A_bar:  torch.Tensor,   # (state_dim,)  diagonal of discrete-time A
    B_bar:  torch.Tensor,   # (state_dim, input_dim)
    C:      torch.Tensor,   # (output_dim, state_dim)
) -> tuple:
    """
    Exact Grammians via the closed-form solution to the discrete Lyapunov equations
    for a diagonal-A system:
      A Wc A^T + B B^T = Wc  ->  Wc_ij = (B B^T)_ij / (1 - a_i * a_j)
      A^T Wo A + C^T C = Wo  ->  Wo_ij = (C^T C)_ij / (1 - a_i * a_j)

    Requires |a_i| < 1 for all i (stable system) -- guaranteed by SSM construction.
    """
    # denominator matrix: denom_ij = 1 - a_i * a_j
    denom = 1.0 - torch.outer(A_bar, A_bar)   # (n, n)  all positive for stable |a_i|<1

    BBT = B_bar @ B_bar.T    # (n, n)
    CTC = C.T @ C            # (n, n)

    Wc = BBT / denom          # (n, n)  symmetric PSD
    Wo = CTC / denom          # (n, n)  symmetric PSD

    return Wc, Wo


def compute_grammians(
    snapshots:  torch.Tensor,   # (NT, state_dim)
    C_matrix:   torch.Tensor,   # (output_dim, state_dim)
) -> tuple:
    """
    Empirical Grammians from hidden-state snapshot trajectories.

    Wc: reachability -- which state directions are excited by inputs.
    Wo: observability -- which state directions contribute to output.

    Wc = H^T H / NT
    Wo = (H @ C^T)^T (H @ C^T) / NT  -- output-energy-weighted state covariance
    """
    NT = snapshots.shape[0]

    Wc = snapshots.T @ snapshots / NT          # (n, n)

    Y   = snapshots @ C_matrix.T              # (NT, output_dim)
    # Project back to state space via C^T to get (state_dim, state_dim) Wo
    Wo  = (C_matrix @ snapshots.T) @ (C_matrix @ snapshots.T).T / NT  # (output_dim, output_dim)
    # Re-project to state_dim via C^T C weighting
    Wo  = C_matrix.T @ (Y.T @ Y / NT) @ C_matrix  # (n, n)

    return Wc, Wo


def symmetrize(W: torch.Tensor) -> torch.Tensor:
    """Enforce exact symmetry (kills floating-point asymmetry from numerical ops)."""
    return (W + W.T) / 2.0


def add_jitter(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Add small diagonal regularisation to ensure positive-definiteness for Cholesky."""
    n = W.shape[0]
    return W + eps * torch.eye(n, dtype=W.dtype, device=W.device)
