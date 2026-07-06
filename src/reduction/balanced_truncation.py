"""
Balanced Truncation (BT) for discrete LTI model order reduction.

Given a stable diagonal-A system (A_bar, B_bar, C) and Grammians (Wc, Wo):

  Step 1: Cholesky factors
    Wc = Lc Lc^T,  Wo = Lo Lo^T

  Step 2: SVD of Hankel proxy
    M = Lc^T @ Lo
    U, sigma, V^T = svd(M)
    sigma = Hankel Singular Values (HSV), descending

  Step 3: Balancing transform
    T     = Lc @ U @ diag(sigma^{-1/2})          [n x n]
    T_inv = diag(sigma^{-1/2}) @ V^T @ Lo^T      [n x n]
    Verify: T_inv @ T = I

  Step 4: Reduced system (truncate to d_hat columns/rows)
    A_hat = T_inv_r @ diag(A_bar) @ T_r   [d_hat x d_hat]  full matrix (not diagonal)
    B_hat = T_inv_r @ B_bar               [d_hat x input_dim]
    C_hat = C @ T_r                       [output_dim x d_hat]

  Error bound: ||G - G_hat||_{H_inf} <= 2 * sum_{i > d_hat} sigma_i

Public API:
  select_rank(hsv, threshold=0.01) -> int
  bt_reduce(A_bar, B_bar, C, Wc, Wo, d_hat=None, hsv_threshold=0.01)
    -> (A_hat, B_hat, C_hat, hsv, T, T_inv)
"""

import torch
from src.reduction.grammians import symmetrize, add_jitter


def select_rank(hsv: torch.Tensor, threshold: float = 0.01) -> int:
    """
    Select the smallest d_hat such that the retained modes capture
    (1 - threshold) fraction of the total HSV energy.

    Equivalently: keep modes where sigma_i / sigma_1 >= threshold.
    Always returns at least 1.
    """
    normalized = hsv / hsv[0].clamp(min=1e-12)
    d_hat = int((normalized >= threshold).sum().item())
    return max(1, d_hat)


def bt_reduce(
    A_bar:         torch.Tensor,   # (n,) diagonal of discrete A
    B_bar:         torch.Tensor,   # (n, m)
    C:             torch.Tensor,   # (p, n)
    Wc:            torch.Tensor,   # (n, n) reachability Grammian
    Wo:            torch.Tensor,   # (n, n) observability Grammian
    d_hat:         int   = None,
    hsv_threshold: float = 0.01,
) -> tuple:
    """
    Perform Balanced Truncation and return reduced matrices.

    Returns:
      A_hat  : (d_hat, d_hat)  full matrix (no longer diagonal)
      B_hat  : (d_hat, m)
      C_hat  : (p, d_hat)
      hsv    : (n,)  Hankel Singular Values (descending)
      T      : (n, n) balancing transform
      T_inv  : (n, n) inverse balancing transform
    """
    # BT uses Cholesky and SVD -- move everything to CPU for MPS compatibility
    # (this is an offline offline step, so CPU speed is fine)
    cpu = torch.device("cpu")
    A_bar = A_bar.to(cpu).float()
    B_bar = B_bar.to(cpu).float()
    C     = C.to(cpu).float()
    Wc    = Wc.to(cpu).float()
    Wo    = Wo.to(cpu).float()

    dtype  = A_bar.dtype
    n      = A_bar.shape[0]

    # Symmetrize and regularize for numerical Cholesky stability
    Wc = add_jitter(symmetrize(Wc))
    Wo = add_jitter(symmetrize(Wo))

    # Step 1: Cholesky factors  Wc = Lc Lc^T,  Wo = Lo Lo^T
    Lc = torch.linalg.cholesky(Wc)   # (n, n) lower triangular
    Lo = torch.linalg.cholesky(Wo)   # (n, n) lower triangular

    # Step 2: SVD of Hankel proxy  M = Lc^T @ Lo
    M = Lc.T @ Lo                    # (n, n)
    U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
    # sigma is descending (torch.linalg.svd guarantees this)

    # Hankel Singular Values
    hsv = sigma.clamp(min=0.0)       # numerical guard

    # Select truncation rank
    if d_hat is None:
        d_hat = select_rank(hsv, hsv_threshold)
    d_hat = min(d_hat, n)

    # Step 3: Balancing transform
    sigma_sqrt_inv = 1.0 / torch.sqrt(sigma.clamp(min=1e-12))   # (n,)

    # T[i,j] = (Lc @ U)[i,j] * sigma_sqrt_inv[j]   -- column j scaled
    T     = (Lc @ U) * sigma_sqrt_inv.unsqueeze(0)              # (n, n)

    # T_inv[i,j] = sigma_sqrt_inv[i] * (Vt @ Lo^T)[i,j]  -- row i scaled
    T_inv = sigma_sqrt_inv.unsqueeze(1) * (Vt @ Lo.T)           # (n, n)

    # Truncate: keep only the first d_hat modes
    T_r     = T[:, :d_hat]      # (n, d_hat)
    T_inv_r = T_inv[:d_hat, :]  # (d_hat, n)

    # Step 4: Reduced system
    # diag(A_bar) @ T_r = A_bar[:, None] * T_r  (row-wise scale of T_r)
    A_hat = T_inv_r @ (A_bar.unsqueeze(1) * T_r)   # (d_hat, d_hat)  full matrix
    B_hat = T_inv_r @ B_bar                          # (d_hat, m)
    C_hat = C @ T_r                                  # (p, d_hat)

    return A_hat, B_hat, C_hat, hsv, T, T_inv


def bt_error_bound(hsv: torch.Tensor, d_hat: int) -> float:
    """
    H-infinity error bound for truncation at d_hat:
      ||G - G_hat||_{H_inf} <= 2 * sum_{i > d_hat} sigma_i
    """
    return 2.0 * float(hsv[d_hat:].sum().item())
