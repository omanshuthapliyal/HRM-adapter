"""
Stability utilities for discrete-time SSM matrices.

check_stability(A) -> bool
  True if all eigenvalues of A satisfy |lambda| < 1.
  Accepts (n, n) full matrix or (n,) diagonal vector.

project_stable(A, eps=1e-6) -> A_stable
  Rescales eigenvalues outside the unit circle to (1 - eps) while preserving
  the eigenvector structure.  Returns real-valued matrix.

plot_hsv(hsv, threshold=0.01, d_hat=None, save_path=None)
  Plots HSV decay curve with threshold line and selected rank marked.
  Saves to save_path if given; otherwise writes to logs/hsv_decay.png.
"""

import os
import torch


def check_stability(A: torch.Tensor) -> bool:
    """
    A: (n,) diagonal vector or (n, n) full matrix.
    Returns True if all discrete-time eigenvalues satisfy |lambda| < 1.
    """
    if A.dim() == 1:
        # Diagonal system -- eigenvalues are the entries themselves
        return bool((A.abs() < 1.0).all().item())
    # Full matrix -- compute eigenvalues
    eigvals = torch.linalg.eigvals(A.to(torch.complex64))
    return bool((eigvals.abs() < 1.0).all().item())


def project_stable(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Project A to be stable: eigenvalues with |lambda| >= 1 are rescaled to (1 - eps).

    For a diagonal (n,) vector: clamp abs values directly.
    For a full (n, n) matrix:   eigendecompose -> rescale -> reconstruct (real part).
    """
    if A.dim() == 1:
        # Diagonal: rescale any |a_i| >= 1 to (1-eps) * sign(a_i)
        magnitudes = A.abs().clamp(min=1.0)          # >= 1 for unstable entries
        return A * (1.0 - eps) / magnitudes          # divides only those >= 1

    # Full matrix
    A_c = A.to(torch.complex64)
    eigvals, V = torch.linalg.eig(A_c)

    # Rescale unstable eigenvalues
    magnitudes  = eigvals.abs()
    scale       = torch.where(magnitudes >= 1.0,
                              (1.0 - eps) / magnitudes.clamp(min=1e-12),
                              torch.ones_like(magnitudes))
    eigvals_stable = eigvals * scale

    # Reconstruct: A_stable = V @ diag(lambda_stable) @ V^{-1}
    V_inv      = torch.linalg.inv(V)
    A_stable_c = V @ torch.diag(eigvals_stable) @ V_inv
    return A_stable_c.real.to(A.dtype)


def plot_hsv(
    hsv:        torch.Tensor,
    threshold:  float = 0.01,
    d_hat:      int   = None,
    save_path:  str   = None,
) -> None:
    """
    Plot normalized HSV decay curve.

    The threshold line marks sigma_i / sigma_1 = threshold.
    d_hat (if given) is marked with a vertical line.
    Saves to save_path; defaults to logs/hsv_decay.png if None.
    """
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe on headless/Mac)
    import matplotlib.pyplot as plt

    hsv_np = hsv.detach().cpu().float().numpy()
    normalized = hsv_np / max(hsv_np[0], 1e-12)
    ranks = list(range(1, len(hsv_np) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(ranks, normalized, "o-", color="steelblue", markersize=4, label="HSV")
    ax.axhline(threshold, color="firebrick", linestyle="--",
               label=f"threshold={threshold}")

    if d_hat is not None:
        ax.axvline(d_hat, color="darkorange", linestyle=":",
                   label=f"d_hat={d_hat}")

    ax.set_xlabel("Mode index")
    ax.set_ylabel("sigma_i / sigma_1  (log scale)")
    ax.set_title("Hankel Singular Value Decay")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    if save_path is None:
        os.makedirs("logs", exist_ok=True)
        save_path = "logs/hsv_decay.png"

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[stability] HSV plot saved to {save_path}")
