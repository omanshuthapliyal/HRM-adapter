"""
LoRA baseline training script -- two-phase execution.

Phase 1 (pre-train):
  Train the full TinyGPT on the copy task with all parameters free.
  Gate: val_acc >= 90%.

Phase 2 (lora-finetune):
  Load pre-trained checkpoint, freeze base model, inject LoRA, fine-tune on
  any task from the registry.  Reports base accuracy before injection.

Usage:
  python scripts/train_lora_baseline.py --phase pretrain
  python scripts/train_lora_baseline.py --phase finetune --load checkpoints/base/best.pt --task parity
  python scripts/train_lora_baseline.py --phase finetune --load checkpoints/base/best.pt --task cumsum_mod
  python scripts/train_lora_baseline.py --phase finetune --load checkpoints/base/best.pt --task delay_copy
  python scripts/train_lora_baseline.py --phase finetune --load checkpoints/base/best.pt --task selective_copy
  python scripts/train_lora_baseline.py --phase finetune --load checkpoints/base/best.pt --task ar
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import TinyGPT, GPTConfig
from src.adapters.insertion import inject_lora
from src.data.copy_task import make_copy_loaders
from src.data.associative_recall import make_ar_loaders
from src.data.recurrent_tasks import TASK_REGISTRY, make_delay_copy_loaders, \
    make_selective_copy_loaders, make_parity_loaders, make_cumsum_mod_loaders, \
    make_parity_last_loaders, make_cumsum_last_loaders, make_majority_loaders
from src.data.utils import seed_everything
from src.training.trainer import Trainer, make_cosine_scheduler
from src.training.metrics import parameter_count
from src.utils.logging import Logger

_VOCAB_SIZE = 32


def build_model(vocab_size: int = _VOCAB_SIZE) -> tuple:
    cfg = GPTConfig(
        d_model=128, n_heads=4, n_layers=2, d_ff=512,
        seq_len=64, vocab_size=vocab_size, dropout=0.0,
    )
    return TinyGPT(cfg), cfg


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_task_data(args, seed_offset: int = 0):
    """Return (train_loader, val_loader, chance_acc) for the chosen task."""
    seed = args.seed + seed_offset
    task = args.task

    if task == "copy":
        loaders = make_copy_loaders(
            copy_len=args.copy_len, vocab_size=16,
            n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = 1.0 / 14
    elif task == "ar":
        loaders = make_ar_loaders(
            n_pairs=args.n_pairs, n_keys=args.n_keys, n_values=args.n_values,
            vocab_size=_VOCAB_SIZE,
            n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = 1.0 / args.n_values
    elif task == "delay_copy":
        loaders = make_delay_copy_loaders(
            n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = TASK_REGISTRY["delay_copy"]["chance"]
    elif task == "selective_copy":
        loaders = make_selective_copy_loaders(
            n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = TASK_REGISTRY["selective_copy"]["chance"]
    elif task == "parity":
        loaders = make_parity_loaders(
            n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = TASK_REGISTRY["parity"]["chance"]
    elif task == "cumsum_mod":
        loaders = make_cumsum_mod_loaders(
            n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = TASK_REGISTRY["cumsum_mod"]["chance"]
    elif task == "parity_last":
        loaders = make_parity_last_loaders(
            n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = TASK_REGISTRY["parity_last"]["chance"]
    elif task == "cumsum_last":
        loaders = make_cumsum_last_loaders(
            n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = TASK_REGISTRY["cumsum_last"]["chance"]
    elif task == "majority":
        loaders = make_majority_loaders(
            n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = TASK_REGISTRY["majority"]["chance"]
    else:
        raise ValueError(f"Unknown task: {task}")

    return loaders[0], loaders[1], chance


# -----------------------------------------------------------------------
# Phase 1: pre-train full model on Copy Task
# -----------------------------------------------------------------------

def phase_pretrain(args):
    seed_everything(args.seed)
    device = get_device()
    print(f"[pretrain] device={device}")

    model, cfg = build_model(vocab_size=_VOCAB_SIZE)
    model = model.to(device)
    print(f"[pretrain] {parameter_count(model)}")

    train_loader, val_loader, _ = load_task_data(
        argparse.Namespace(
            task="copy", seed=args.seed, copy_len=args.copy_len,
            n_train=args.n_train, n_val=args.n_val, batch=args.batch,
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = min(100, total_steps // 10)
    scheduler = make_cosine_scheduler(optimizer, warmup_steps, total_steps)

    logger = Logger(args.log_dir, run_name="pretrain")
    logger.log_config(vars(args))

    trainer = Trainer(
        model, optimizer, scheduler,
        train_loader, val_loader,
        {"epochs": args.epochs, "grad_clip": 1.0,
         "save_every": 10, "save_dir": "checkpoints/base"},
        logger,
    )
    trainer.train()

    final = trainer.eval()
    print(f"\n[pretrain] Final val_acc={final['accuracy']:.4f}")
    gate = final["accuracy"] >= 0.90
    print(f"Step gate ({'PASS' if gate else 'FAIL'}): val_acc >= 0.90")
    if not gate:
        sys.exit(1)


# -----------------------------------------------------------------------
# Phase 2: inject LoRA, freeze base, fine-tune on chosen task
# -----------------------------------------------------------------------

def phase_finetune(args):
    if not args.load:
        print("ERROR: --load <checkpoint_path> required for finetune phase")
        sys.exit(1)

    seed_everything(args.seed + 999)
    device = get_device()
    print(f"[lora-finetune] device={device}  task={args.task}")

    model, cfg = build_model(vocab_size=_VOCAB_SIZE)
    model = model.to(device)
    ckpt = torch.load(args.load, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[lora-finetune] Loaded base checkpoint: {args.load}")

    train_loader, val_loader, chance = load_task_data(args, seed_offset=999)

    # Eval base BEFORE LoRA
    print(f"\n[lora-finetune] Base model on {args.task} (before LoRA)...")
    _dummy_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    _eval = Trainer(
        model, _dummy_opt, None, train_loader, val_loader,
        {"epochs": 0, "grad_clip": 0.0, "save_every": 9999,
         "save_dir": "/tmp/_noop_lora_base"},
        Logger(args.log_dir, run_name=f"lora_base_{args.task}"),
    )
    base = _eval.eval()
    print(f"[lora-finetune] Base: loss={base['loss']:.4f}  "
          f"acc={base['accuracy']:.4f}  (chance={chance:.4f})")

    # Inject LoRA
    lora_cfg = {"lora": {
        "rank": args.rank, "alpha": args.alpha,
        "target_modules": ["attn.q_proj", "attn.v_proj"],
    }}
    model = inject_lora(model, lora_cfg)
    model = model.to(device)
    print(f"[lora-finetune] {parameter_count(model)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr * 0.3, weight_decay=0.0,
    )
    total_steps  = args.finetune_epochs * len(train_loader)
    warmup_steps = min(50, total_steps // 10)
    scheduler = make_cosine_scheduler(optimizer, warmup_steps, total_steps)

    logger = Logger(args.log_dir, run_name=f"lora_ft_{args.task}")
    logger.log_config(vars(args))

    trainer = Trainer(
        model, optimizer, scheduler, train_loader, val_loader,
        {"epochs": args.finetune_epochs, "grad_clip": 1.0,
         "save_every": 5, "save_dir": f"checkpoints/lora_{args.task}"},
        logger,
    )
    trainer.train()

    final = trainer.eval()
    print(f"\n[lora-finetune] Final: acc={final['accuracy']:.4f}  "
          f"(base={base['accuracy']:.4f}  delta={final['accuracy']-base['accuracy']:+.4f})")


# -----------------------------------------------------------------------

def parse_args():
    TASKS = ["copy", "ar", "delay_copy", "selective_copy", "parity", "cumsum_mod",
             "parity_last", "cumsum_last", "majority"]
    p = argparse.ArgumentParser()
    p.add_argument("--phase",           choices=["pretrain", "finetune"], default="pretrain")
    p.add_argument("--task",            choices=TASKS, default="parity")
    p.add_argument("--load",            type=str,   default=None)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--finetune_epochs", type=int,   default=20)
    p.add_argument("--rank",            type=int,   default=8)
    p.add_argument("--alpha",           type=float, default=16.0)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--batch",           type=int,   default=64)
    p.add_argument("--copy_len",        type=int,   default=16)
    p.add_argument("--n_train",         type=int,   default=5_000)
    p.add_argument("--n_val",           type=int,   default=500)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--log_dir",         type=str,   default="logs")
    p.add_argument("--n_pairs",         type=int,   default=4)
    p.add_argument("--n_keys",          type=int,   default=16)
    p.add_argument("--n_values",        type=int,   default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "pretrain":
        phase_pretrain(args)
    else:
        phase_finetune(args)
