"""
From-scratch sweep training for long-horizon tasks.

Trains TinyGPT (+ optional adapter) from random init on a single task at a given
sequence length. ALL parameters are trainable -- no frozen base, no pretrained checkpoint.
This is "Option A" for the parity sweep and DFA experiments.

Adapter modes:
  none  -- plain TinyGPT, full model trained
  lora  -- inject LoRA into Q,V, then unfreeze all params (LoRA adds low-rank bottleneck)
  hrm   -- inject HRM SSM adapter, then unfreeze all params (SSM adds recurrent state)

Checkpoint saved to: checkpoints/sweep_{task}_T{T}_{adapter}/best.pt
Log CSV saved to:    logs/sweep_{task}_T{T}_{adapter}.csv

The HRM checkpoint format is compatible with reduce_adapter.py (pass --seq_len / --vocab_size).

Usage:
  python scripts/train_sweep.py --task long_parity --T 64  --adapter lora
  python scripts/train_sweep.py --task long_parity --T 256 --adapter hrm
  python scripts/train_sweep.py --task dfa         --T 128 --adapter lora
  python scripts/train_sweep.py --task dfa         --T 512 --adapter hrm
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import TinyGPT, GPTConfig
from src.adapters.insertion import inject_lora, inject_hrm
from src.adapters.hrm_adapter import HRMAdapter
from src.data.long_range_tasks import (
    make_long_parity_loaders, PARITY_VOCAB, PARITY_SEP,
    make_dfa_loaders, DFA_VOCAB, dfa_vocab_size, dfa_seq_len,
)
from src.data.maestro import make_maestro_loaders, MAESTRO_VOCAB_SIZE
from src.data.enwiki8 import make_enwiki8_loaders, ENWIKI8_VOCAB_SIZE
from src.data.utils import seed_everything
from src.training.trainer import Trainer, make_cosine_scheduler
from src.training.metrics import parameter_count
from src.utils.logging import Logger


# -----------------------------------------------------------------------
# Task specs: (vocab_size, seq_len_fn)
# seq_len_fn(T) returns minimum model seq_len for T-length task
# -----------------------------------------------------------------------
TASK_SPECS = {
    "long_parity": {
        "vocab_size": PARITY_VOCAB,
        "seq_len_fn": lambda T, **kw: 2 * T + 1,
        "loader_fn":  make_long_parity_loaders,
        "chance":     0.5,
    },
    "dfa": {
        # vocab_size and seq_len are n_states-dependent; resolved in main() via args.n_states
        "vocab_size": None,
        "seq_len_fn": None,
        "loader_fn":  make_dfa_loaders,
        "chance":     None,
    },
    "maestro": {
        # vocab_size resolved at runtime from miditok tokenizer; seq_len = args.T
        "vocab_size": None,
        "seq_len_fn": None,
        "loader_fn":  None,   # handled explicitly in main()
        "chance":     None,
    },
    "enwiki8": {
        "vocab_size": ENWIKI8_VOCAB_SIZE,
        "seq_len_fn": lambda T, **kw: T,
        "loader_fn":  make_enwiki8_loaders,
        "chance":     1.0 / ENWIKI8_VOCAB_SIZE,
    },
}

LORA_TARGET_MODULES = ["attn.q_proj", "attn.v_proj"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(seq_len: int, vocab_size: int, args=None) -> tuple:
    d_model  = getattr(args, 'd_model',  128)
    n_heads  = getattr(args, 'n_heads',  4)
    n_layers = getattr(args, 'n_layers', 2)
    d_ff     = getattr(args, 'd_ff',     512)
    dropout  = getattr(args, 'dropout',  0.1)
    cfg = GPTConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        seq_len=seq_len, vocab_size=vocab_size, dropout=dropout,
    )
    return TinyGPT(cfg), cfg


def main(args):
    seed_everything(args.seed)
    device = get_device()

    spec = TASK_SPECS[args.task]

    # Resolve task-specific parameters
    if args.task == "dfa":
        vocab_size = dfa_vocab_size(args.n_states)
        seq_len    = dfa_seq_len(args.T, args.n_states)
        chance     = 1.0 / args.n_states
    elif args.task == "maestro":
        # vocab_size is determined by miditok at loader construction time;
        # use MAESTRO_VOCAB_SIZE as a safe default for model building.
        vocab_size = MAESTRO_VOCAB_SIZE
        seq_len    = args.T            # --T is the context window length
        chance     = 1.0 / vocab_size
    elif args.task == "enwiki8":
        vocab_size = ENWIKI8_VOCAB_SIZE
        seq_len    = args.T
        chance     = 1.0 / vocab_size
    else:
        vocab_size = spec["vocab_size"]
        seq_len    = 2 * args.T + 1   # long_parity
        chance     = spec["chance"]

    suffix   = f"_{args.run_suffix}" if args.run_suffix else ""
    run_name = f"sweep_{args.task}_T{args.T}_{args.adapter}{suffix}"
    print(f"[sweep] task={args.task}  T={args.T}  adapter={args.adapter}  n_states={args.n_states}")
    print(f"[sweep] seq_len={seq_len}  vocab_size={vocab_size}  device={device}")

    # --- Build model ---
    model, cfg = build_model(seq_len, vocab_size, args)
    model = model.to(device)

    hrm_cfg = None

    if args.adapter == "lora":
        lora_cfg = {
            "lora": {
                "rank":           args.lora_rank,
                "alpha":          float(args.lora_rank * 2),
                "target_modules": LORA_TARGET_MODULES,
            }
        }
        model = inject_lora(model, lora_cfg)
        model = model.to(device)   # move LoRA params (lora_A, lora_B) to device
        # Unfreeze ALL params -- from-scratch training, not adapter-only fine-tuning
        for p in model.parameters():
            p.requires_grad_(True)

    elif args.adapter == "hrm":
        hrm_cfg = {
            "hrm": {
                "state_dim":  args.state_dim,
                "input_dim":  cfg.d_model,
                "output_dim": cfg.d_model,
                "dt_init":    0.01,
                "dt_min":     1e-4,
                "dt_max":     0.1,
                "gate_init":  args.gate_init,
            }
        }
        model = inject_hrm(model, hrm_cfg)
        model = model.to(device)
        # Unfreeze ALL params -- from-scratch training
        for p in model.parameters():
            p.requires_grad_(True)

    # else: args.adapter == "none" -- plain TinyGPT, all params already trainable

    params = parameter_count(model)
    print(f"[sweep] {params}")

    # --- Data ---
    if args.task == "maestro":
        train_loader, val_loader = make_maestro_loaders(
            data_dir=args.maestro_dir,
            seq_len=seq_len,
            n_train=args.n_train,
            n_val=args.n_val,
            batch_size=args.batch,
            seed=args.seed,
        )
        # Update vocab_size to the actual tokenizer vocab (may differ from default)
        vocab_size = train_loader.dataset.dataset.vocab_size if hasattr(
            train_loader.dataset, "dataset") else train_loader.dataset.vocab_size
        chance = 1.0 / vocab_size
    elif args.task == "enwiki8":
        train_loader, val_loader = make_enwiki8_loaders(
            data_dir=args.data_dir,
            seq_len=seq_len,
            n_train=args.n_train,
            n_val=args.n_val,
            batch_size=args.batch,
            seed=args.seed,
        )
    else:
        loader_kwargs = dict(T=args.T, n_train=args.n_train, n_val=args.n_val,
                             batch_size=args.batch, seed=args.seed)
        if args.task == "dfa":
            loader_kwargs["n_states"] = args.n_states
        train_loader, val_loader = spec["loader_fn"](**loader_kwargs)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)
    scheduler    = make_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # --- Logger ---
    logger = Logger(args.log_dir, run_name=run_name)
    logger.log_config(vars(args))

    save_dir = os.path.join("checkpoints", run_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- Train ---
    trainer = Trainer(
        model, optimizer, scheduler,
        train_loader, val_loader,
        {
            "epochs":     args.epochs,
            "grad_clip":  1.0,
            "save_every": max(5, args.epochs // 5),
            "save_dir":   save_dir,
        },
        logger,
    )
    trainer.train()

    # --- Final eval ---
    # Use best_val_acc (from saved best.pt checkpoint) as the canonical result.
    # Last-epoch accuracy can be lower due to oscillation; best is the fair metric.
    import math as _math
    metrics = trainer.eval()
    best_val_acc = trainer.best_val_acc
    # BPC = cross-entropy (nats) / ln(2); computed from best checkpoint's loss
    best_bpc = trainer.best_val_loss / _math.log(2)
    bpc_str  = f"  val_bpc={best_bpc:.4f}" if args.task == "enwiki8" else ""
    print(f"\n[sweep] Final  val_acc={metrics['accuracy']:.4f}  "
          f"best_val_acc={best_val_acc:.4f}  "
          f"val_loss={metrics['loss']:.4f}  chance={chance:.4f}{bpc_str}")

    # --- Save checkpoint compatible with reduce_adapter.py ---
    out = {
        "model_state_dict": model.state_dict(),
        "best_val_acc":     best_val_acc,
        "global_step":      trainer.global_step,
        "model_config": {
            "seq_len":    seq_len,
            "vocab_size": vocab_size,
            "d_model":    cfg.d_model,
            "n_heads":    cfg.n_heads,
            "n_layers":   cfg.n_layers,
            "d_ff":       cfg.d_ff,
        },
    }
    if hrm_cfg is not None:
        out["hrm_config"] = hrm_cfg

    ckpt_path = os.path.join(save_dir, "best.pt")
    torch.save(out, ckpt_path)
    print(f"[sweep] Checkpoint saved to {ckpt_path}")
    result_line = f"[sweep] RESULT  {run_name}  val_acc={best_val_acc:.4f}{bpc_str}"
    print(result_line)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",        choices=list(TASK_SPECS.keys()), required=True)
    p.add_argument("--maestro_dir", type=str, default="data/maestro-v3.0.0",
                   help="Path to extracted MAESTRO v3 dataset (only used for --task maestro)")
    p.add_argument("--data_dir",    type=str, default="data/enwiki8",
                   help="Path to dataset directory (used for --task enwiki8)")
    p.add_argument("--T",          type=int,   required=True,
                   help="Parity bits T, or DFA input length T")
    p.add_argument("--adapter",    choices=["none", "lora", "hrm"], default="lora")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-3)
    p.add_argument("--n_train",    type=int,   default=5_000)
    p.add_argument("--n_val",      type=int,   default=500)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--lora_rank",  type=int,   default=8)
    p.add_argument("--state_dim",  type=int,   default=32)
    p.add_argument("--gate_init",  type=float, default=0.01,
                   help="Initial gate scalar for HRM adapter")
    p.add_argument("--n_states",   type=int,   default=4,
                   help="Number of DFA states (only used for --task dfa)")
    p.add_argument("--run_suffix", type=str,   default="",
                   help="Suffix appended to checkpoint/run name (e.g. seed123)")
    p.add_argument("--log_dir",    type=str,   default="logs")
    # Backbone architecture overrides (for GPU experiments with larger models)
    p.add_argument("--d_model",    type=int,   default=128)
    p.add_argument("--n_heads",    type=int,   default=4)
    p.add_argument("--n_layers",   type=int,   default=2)
    p.add_argument("--d_ff",       type=int,   default=512)
    p.add_argument("--dropout",    type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
