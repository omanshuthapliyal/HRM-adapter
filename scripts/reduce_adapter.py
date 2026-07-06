"""
Reduction script: load phase1 HRM checkpoint -> compute Grammians -> BT reduce -> save.

For each HRMAdapter layer (2 layers for the default TinyGPT):
  1. Extract trained A_bar, B_bar, C from the SSM
  2. Compute exact Grammians (closed-form for diagonal A)
  3. Balanced Truncation -> (A_hat, B_hat, C_hat, HSV)
  4. Check stability of A_hat; project if needed
  5. Call adapter.replace_ssm_matrices(A_hat, B_hat, C_hat)

Saves:
  checkpoints/hrm_reduced_{task}/model_reduced.pt   -- full model state + reduced matrices
  logs/hsv_decay_{task}_layer{i}.png                -- HSV decay plot per layer

Usage:
  python scripts/reduce_adapter.py --load checkpoints/hrm_phase1_ar/best.pt --task ar
  python scripts/reduce_adapter.py --load checkpoints/hrm_phase1_parity/best.pt --task parity
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import TinyGPT, GPTConfig
from src.adapters.insertion import inject_hrm
from src.adapters.hrm_adapter import HRMAdapter
from src.reduction.grammians import compute_exact_grammians
from src.reduction.balanced_truncation import bt_reduce, select_rank, bt_error_bound
from src.reduction.stability import check_stability, project_stable, plot_hsv

_VOCAB_SIZE = 32
_SEQ_LEN    = 64


def build_model(vocab_size: int = _VOCAB_SIZE, seq_len: int = _SEQ_LEN) -> tuple:
    cfg = GPTConfig(
        d_model=128, n_heads=4, n_layers=2, d_ff=512,
        seq_len=seq_len, vocab_size=vocab_size, dropout=0.0,
    )
    return TinyGPT(cfg), cfg


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(args):
    if not args.load:
        print("ERROR: --load <checkpoint_path> required")
        sys.exit(1)

    device = get_device()
    print(f"[reduce] device={device}")

    # Resolve model config: prefer values embedded in checkpoint, fall back to CLI args
    _raw = torch.load(args.load, map_location="cpu", weights_only=False)
    _mc  = _raw.get("model_config", {})
    # CLI flags take precedence over embedded model_config (handles checkpoints with stale config)
    vocab_size = args.vocab_size if args.vocab_size != _VOCAB_SIZE else _mc.get("vocab_size", _VOCAB_SIZE)
    seq_len    = args.seq_len    if args.seq_len    != _SEQ_LEN    else _mc.get("seq_len",    _SEQ_LEN)
    hrm_cfg_ckpt = _raw.get("hrm_config", None)

    # Build model and inject HRM (same config as training)
    model, cfg = build_model(vocab_size=vocab_size, seq_len=seq_len)
    model = model.to(device)

    # Use HRM config from checkpoint if present (sweep runs embed it), else build from args
    if hrm_cfg_ckpt is not None:
        hrm_cfg = hrm_cfg_ckpt
        hrm_cfg["hrm"]["input_dim"]  = cfg.d_model
        hrm_cfg["hrm"]["output_dim"] = cfg.d_model
    else:
        hrm_cfg = {
            "hrm": {
                "state_dim":  args.state_dim,
                "input_dim":  cfg.d_model,
                "output_dim": cfg.d_model,
                "dt_init":    0.01,
                "dt_min":     1e-4,
                "dt_max":     0.1,
            }
        }
    model = inject_hrm(model, hrm_cfg)
    model = model.to(device)

    # Load trained weights (works for both frozen-base and from-scratch checkpoints)
    ckpt = torch.load(args.load, map_location=device, weights_only=False)
    # From-scratch checkpoints have all params trained; load with strict=False to handle
    # any shape differences that arise if gate_init key is missing from older checkpoints
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"[reduce] WARNING: missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[reduce] WARNING: unexpected keys in state_dict: {unexpected}")
    print(f"[reduce] Loaded phase1 checkpoint: {args.load}")

    # Collect all HRMAdapter modules
    hrm_layers = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, HRMAdapter)
    ]
    print(f"[reduce] Found {len(hrm_layers)} HRMAdapter layer(s)")

    save_dir = f"checkpoints/hrm_reduced_{args.task}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    reduction_meta = {}    # layer_name -> {d_hat, hsv, error_bound}

    for layer_idx, (name, adapter) in enumerate(hrm_layers):
        print(f"\n[reduce] Layer '{name}'  (state_dim={adapter.ssm.state_dim})")

        # Extract trained discrete SSM matrices
        A_bar, B_bar, C = adapter.ssm.get_matrices()   # all detached CPU tensors
        A_bar = A_bar.to(device)
        B_bar = B_bar.to(device)
        C     = C.to(device)

        print(f"  A_bar range: [{A_bar.min().item():.4f}, {A_bar.max().item():.4f}]")
        print(f"  |gate|={adapter.gate.abs().item():.6f}")

        # Compute exact Grammians (analytic, no approximation for diagonal A)
        Wc, Wo = compute_exact_grammians(A_bar, B_bar, C)
        print(f"  Wc diag range: [{Wc.diag().min().item():.2e}, {Wc.diag().max().item():.2e}]")
        print(f"  Wo diag range: [{Wo.diag().min().item():.2e}, {Wo.diag().max().item():.2e}]")

        # Balanced Truncation
        A_hat, B_hat, C_hat, hsv, T, T_inv = bt_reduce(
            A_bar, B_bar, C, Wc, Wo,
            d_hat=None,
            hsv_threshold=args.hsv_threshold,
        )
        d_hat = A_hat.shape[0]
        error = bt_error_bound(hsv, d_hat)

        print(f"  HSV (top-5): {hsv[:5].tolist()}")
        print(f"  Truncation: {args.state_dim} -> {d_hat}  "
              f"(threshold={args.hsv_threshold})")
        print(f"  H-inf error bound: {error:.6f}")

        # Stability check on reduced A_hat
        stable = check_stability(A_hat)
        print(f"  A_hat stable: {stable}")
        if not stable:
            print("  WARNING: A_hat unstable -- projecting to stable region")
            A_hat = project_stable(A_hat)
            assert check_stability(A_hat), "project_stable failed"
            print("  A_hat projected -- now stable")

        # HSV decay plot
        plot_hsv(
            hsv, threshold=args.hsv_threshold, d_hat=d_hat,
            save_path=f"logs/hsv_decay_{args.task}_layer{layer_idx}.png",
        )

        # Hot-swap SSM matrices with reduced versions
        adapter.replace_ssm_matrices(A_hat, B_hat, C_hat)

        reduction_meta[name] = {
            "d_hat":        d_hat,
            "d_orig":       args.state_dim,
            "hsv":          hsv.cpu(),
            "error_bound":  error,
            "A_hat":        A_hat.cpu(),
            "B_hat":        B_hat.cpu(),
            "C_hat":        C_hat.cpu(),
        }

    # Summary
    print("\n[reduce] Reduction summary:")
    for name, meta in reduction_meta.items():
        ratio = meta["d_hat"] / meta["d_orig"]
        print(f"  {name}: {meta['d_orig']} -> {meta['d_hat']}  "
              f"({ratio:.1%} of original)  "
              f"H-inf bound={meta['error_bound']:.4f}")

    # Save reduced model -- include model_config so phase2 can rebuild correctly
    out = {
        "model_state_dict":  ckpt["model_state_dict"],
        "reduction_meta":    reduction_meta,
        "hrm_config":        hrm_cfg,
        "model_config":      ckpt.get("model_config", {"seq_len": seq_len, "vocab_size": vocab_size}),
        "global_step":       ckpt.get("global_step", 0),
        "best_val_acc":      ckpt.get("best_val_acc", 0.0),
    }
    out_path = os.path.join(save_dir, "model_reduced.pt")
    torch.save(out, out_path)
    print(f"\n[reduce] Saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load",          type=str,   default=None,
                   help="HRM checkpoint path (phase1 or sweep)")
    p.add_argument("--task",          type=str,   default="ar",
                   help="Task name -- used for output directory naming")
    p.add_argument("--state_dim",     type=int,   default=32)
    p.add_argument("--hsv_threshold", type=float, default=0.01,
                   help="Keep modes with sigma_i/sigma_1 >= threshold")
    p.add_argument("--seq_len",       type=int,   default=_SEQ_LEN,
                   help="Model seq_len (overridden by checkpoint model_config if present)")
    p.add_argument("--vocab_size",    type=int,   default=_VOCAB_SIZE,
                   help="Vocab size (overridden by checkpoint model_config if present)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
