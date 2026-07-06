#!/usr/bin/env python3
"""
Balanced Truncation reduction of trained Mistral-7B HRM adapters.

For each taskxseed checkpoint:
  1. Load trained SSM params (log_A, B, C, log_dt) for all 32 layers
  2. ZOH-discretize -> A_bar, B_bar
  3. Compute exact analytic Grammians (diagonal-A form)
  4. Balanced truncation: Cholesky -> SVD -> truncate at threshold eps
  5. Save reduced matrices and d_hat per layer to logs/bt_reduced_{task}_s{seed}_eps{eps}.pt

Usage (from mistral7b_local/):
    python scripts/reduce_mistral_hrm.py --task quality --seed 42 --eps 0.10

The saved .pt file is consumed by train_peft_longbench.py --bt_reduce_init.

Output file keys per layer i:
    reduced['d_hat']         : list[int] of length N_LAYERS
    reduced['layers'][i]['A_hat']   : (d_hat_i, d_hat_i) tensor
    reduced['layers'][i]['B_hat']   : (d_hat_i, d_model) tensor
    reduced['layers'][i]['C_hat']   : (d_model, d_hat_i) tensor
    reduced['layers'][i]['T_fwd']   : (d, d_hat_i) -- balancing transform (for warm-start)
    reduced['hsv']           : list[list[float]] HSV per layer (normalized)
    reduced['meta']          : dict with task/seed/eps/d_hat_stats
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# -- Self-contained math (no src/ imports needed) --

def discretize(log_A, B, log_dt, dt_min=1e-4, dt_max=0.1):
    """ZOH discretization for diagonal-A SSM."""
    a = -torch.exp(log_A.float())
    delta = F.softplus(log_dt.float()).clamp(dt_min, dt_max)
    A_bar = torch.exp(a * delta)
    B_bar = ((A_bar - 1) / a).unsqueeze(1) * B.float()   # (d, d_model)
    return A_bar, B_bar


def compute_grammians(A_bar, B_bar, C):
    """Exact analytic Grammians for diagonal A_bar.
    Wc_ij = (B B^T)_ij / (1 - a_i*a_j)
    Wo_ij = (C^T C)_ij / (1 - a_i*a_j)
    """
    denom = 1.0 - torch.outer(A_bar, A_bar)          # (d, d)
    Wc = (B_bar @ B_bar.T) / denom
    Wo = (C.T @ C) / denom
    return Wc, Wo


def bt_reduce(A_bar, B_bar, C, eps: float):
    """
    Balanced truncation at threshold eps.

    Returns:
        A_hat, B_hat, C_hat  : reduced system matrices
        T_fwd                 : (d, d_hat) forward transform (x_hat = T_fwd.T @ x)
        hsv_norm              : normalized HSV (sigma/sigma_1)
        d_hat                 : retained state dimension
    """
    d = A_bar.shape[0]
    d_model = B_bar.shape[1]
    eps_jitter = 1e-8
    I = torch.eye(d, dtype=torch.float32)

    Wc, Wo = compute_grammians(A_bar, B_bar, C)
    Wc = (Wc + Wc.T) / 2 + eps_jitter * I
    Wo = (Wo + Wo.T) / 2 + eps_jitter * I

    Lc = torch.linalg.cholesky(Wc)   # Wc = Lc Lc^T
    Lo = torch.linalg.cholesky(Wo)   # Wo = Lo Lo^T
    M = Lc.T @ Lo                     # Hankel proxy (d, d)

    U, sv, Vh = torch.linalg.svd(M, full_matrices=False)  # sv descending
    sv_norm = sv / sv[0].clamp(min=1e-12)

    d_hat = max(1, int((sv_norm >= eps).sum().item()))

    # Balancing transforms
    # Left:  T_fwd = Lc @ U[:, :d_hat] @ diag(sv[:d_hat]^{-1/2})
    # Right: T_inv = Lo @ Vh[:d_hat, :].T @ diag(sv[:d_hat]^{-1/2})
    scale = sv[:d_hat].clamp(min=1e-12).pow(-0.5)   # (d_hat,)
    T_fwd = Lc @ U[:, :d_hat] * scale.unsqueeze(0)  # (d, d_hat)
    T_inv = Lo @ Vh[:d_hat, :].T * scale.unsqueeze(0)  # (d, d_hat)  -- right transform

    # Reduce: A_hat = T_fwd^T @ diag(A_bar) @ T_inv  [diagonal A -> full A_hat]
    A_hat = T_fwd.T @ (A_bar.unsqueeze(1) * T_inv)  # (d_hat, d_hat)

    # B_hat = T_fwd^T @ B_bar  ->  (d_hat, d_model)
    B_hat = T_fwd.T @ B_bar

    # C_hat = C @ T_inv  ->  (d_model, d_hat)
    # C is (d_model, d) in the SSM convention used here
    C_hat = C @ T_inv  # C: (d_model, d), T_inv: (d, d_hat)

    return A_hat, B_hat, C_hat, T_fwd, sv_norm.tolist(), d_hat


# -- Checkpoint discovery and loading --

def find_checkpoint(log_dir: Path, task: str, seed: int):
    p = log_dir / f"mistral7b_hrm_{task}_s{seed}"
    if not p.exists():
        return None
    shards = sorted(p.glob("model-*.safetensors"))
    single = p / "model.safetensors"
    if shards or single.exists():
        return p
    return None


def load_state(ckpt_path: Path) -> dict:
    shards = sorted(ckpt_path.glob("model-*.safetensors"))
    if not shards:
        shards = [ckpt_path / "model.safetensors"]
    s = {}
    for f in shards:
        s.update(load_file(str(f), device="cpu"))
    return s


# -- Main --

def parse_args():
    p = argparse.ArgumentParser(
        description="BT-reduce trained Mistral-7B HRM adapter checkpoints")
    p.add_argument("--task", required=True,
                   choices=["quality", "qmsum", "narrativeqa"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=0.10,
                   help="HSV threshold eps for truncation (default 0.10)")
    p.add_argument("--log_dir", default="logs",
                   help="Directory containing HRM checkpoint folders")
    p.add_argument("--n_layers", type=int, default=32)
    p.add_argument("--dt_min", type=float, default=1e-4)
    p.add_argument("--dt_max", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    ckpt_path = find_checkpoint(log_dir, args.task, args.seed)

    if ckpt_path is None:
        print(f"[ERROR] No checkpoint found for task={args.task} seed={args.seed} "
              f"in {log_dir}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    state = load_state(ckpt_path)

    reduced_layers = []
    d_hats = []

    print(f"\nRunning BT reduction at eps={args.eps}  ({args.n_layers} layers)\n")
    print(f"{'Layer':>6}  {'d_hat':>6}  {'sigma_1':>8}  {'sigma_tail':>8}  {'status'}")
    print("-" * 50)

    for layer in range(args.n_layers):
        prefix = f"model.layers.{layer}.hrm.ssm."
        if f"{prefix}log_A" not in state:
            print(f"{layer:>6}  {'--':>6}  {'--':>8}  {'--':>8}  no SSM key")
            d_hats.append(0)
            reduced_layers.append(None)
            continue

        log_A  = state[f"{prefix}log_A"]
        B      = state[f"{prefix}B"]
        C      = state[f"{prefix}C"]
        log_dt = state[f"{prefix}log_dt"]

        try:
            A_bar, B_bar = discretize(log_A, B, log_dt, args.dt_min, args.dt_max)
            C_f = C.float()
            A_hat, B_hat, C_hat, T_fwd, hsv_norm, d_hat = bt_reduce(
                A_bar, B_bar, C_f, args.eps)

            d_hats.append(d_hat)
            reduced_layers.append({
                "A_hat": A_hat,
                "B_hat": B_hat,
                "C_hat": C_hat,
                "T_fwd": T_fwd,
                "hsv_norm": hsv_norm,
            })
            sigma_tail = hsv_norm[-1] if hsv_norm else 0.0
            print(f"{layer:>6}  {d_hat:>6}  {1.0:>8.4f}  {sigma_tail:>8.4f}  OK")

        except Exception as e:
            print(f"{layer:>6}  {'ERR':>6}  {'--':>8}  {'--':>8}  {e}")
            d_hats.append(-1)
            reduced_layers.append(None)

    # Summary
    valid = [d for d in d_hats if d > 0]
    print(f"\n{'-'*50}")
    print(f"d_hat summary (eps={args.eps}):  "
          f"min={min(valid)}  median={sorted(valid)[len(valid)//2]}  max={max(valid)}")
    print(f"Mean d_hat = {sum(valid)/len(valid):.1f}  "
          f"(compression: {32-sum(valid)/len(valid):.1f} modes removed on average)")

    # Save
    eps_str = f"{int(args.eps * 100):02d}"   # 0.10 -> "10", 0.05 -> "05"
    out_path = log_dir / f"bt_reduced_{args.task}_s{args.seed}_eps{eps_str}.pt"
    torch.save({
        "d_hat":   d_hats,
        "layers":  reduced_layers,
        "meta": {
            "task":    args.task,
            "seed":    args.seed,
            "eps":     args.eps,
            "n_layers": args.n_layers,
            "d_hat_min":    min(valid) if valid else None,
            "d_hat_median": sorted(valid)[len(valid)//2] if valid else None,
            "d_hat_max":    max(valid) if valid else None,
            "d_hat_mean":   round(sum(valid)/len(valid), 2) if valid else None,
            "ckpt_path":    str(ckpt_path),
        },
    }, out_path)
    print(f"\nSaved -> {out_path}")
    print("Pass this file to train_peft_longbench.py via --bt_reduce_init")


if __name__ == "__main__":
    main()
