"""
HRM Phase 1 training script -- two-task-agnostic execution.

Loads the pre-trained TinyGPT base, injects a full-rank SSM adapter (state_dim=32)
via inject_hrm(), freezes base weights, and fine-tunes on the chosen task.

After training, runs a calibration forward pass over the training split and saves
SSM hidden-state snapshots to checkpoints/hrm_phase1_{task}/snapshots.pt for use
in Step 4 (Balanced Truncation / reduction).

Usage:
  python scripts/train_hrm_phase1.py --load checkpoints/base/best.pt --task ar
  python scripts/train_hrm_phase1.py --load checkpoints/base/best.pt --task parity
  python scripts/train_hrm_phase1.py --load checkpoints/base/best.pt --task cumsum_mod
  python scripts/train_hrm_phase1.py --load checkpoints/base/best.pt --task delay_copy
  python scripts/train_hrm_phase1.py --load checkpoints/base/best.pt --task selective_copy

Step 3 gate:
  val_acc >= 0.07 AND gates non-zero (non-selective SSM ceiling on AR is ~10%)
  snapshots.pt exists with non-zero content
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import TinyGPT, GPTConfig
from src.adapters.insertion import inject_hrm
from src.adapters.hrm_adapter import HRMAdapter
from src.data.associative_recall import make_ar_loaders
from src.data.copy_task import make_copy_loaders
from src.data.recurrent_tasks import TASK_REGISTRY, make_delay_copy_loaders, \
    make_selective_copy_loaders, make_parity_loaders, make_cumsum_mod_loaders, \
    make_parity_last_loaders, make_cumsum_last_loaders, make_majority_loaders
from src.data.utils import seed_everything
from src.training.trainer import Trainer, make_cosine_scheduler
from src.training.metrics import parameter_count
from src.reduction.hooks import StateCollector
from src.utils.logging import Logger

_VOCAB_SIZE = 32   # must match pretrain vocab_size


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


def main(args):
    if not args.load:
        print("ERROR: --load <checkpoint_path> required")
        sys.exit(1)

    seed_everything(args.seed + 999)    # same offset as LoRA finetune -> same data split
    device = get_device()
    print(f"[hrm-phase1] device={device}  task={args.task}")

    # Build model and load pre-trained base
    model, cfg = build_model(vocab_size=_VOCAB_SIZE)
    model = model.to(device)
    ckpt = torch.load(args.load, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[hrm-phase1] Loaded base checkpoint: {args.load}")

    train_loader, val_loader, chance = load_task_data(args, seed_offset=999)

    # Evaluate base BEFORE injecting HRM (sanity check)
    print(f"\n[hrm-phase1] Base model on {args.task} (before HRM injection)...")
    _dummy_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    _eval = Trainer(
        model, _dummy_opt, None,
        train_loader, val_loader,
        {"epochs": 0, "grad_clip": 0.0, "save_every": 9999,
         "save_dir": "/tmp/_noop_hrm_base_eval"},
        Logger(args.log_dir, run_name=f"hrm_base_eval_{args.task}"),
    )
    base_metrics = _eval.eval()
    print(f"[hrm-phase1] Base acc={base_metrics['accuracy']:.4f}  "
          f"(chance={chance:.4f})")

    # Inject HRM adapter -- freezes base, only SSM + gate are trainable
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
    model = model.to(device)    # SSM tensors created on CPU -- move to device
    print(f"\n[hrm-phase1] After inject_hrm: {parameter_count(model)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = min(50, total_steps // 10)
    scheduler    = make_cosine_scheduler(optimizer, warmup_steps, total_steps)

    logger = Logger(args.log_dir, run_name=f"hrm_phase1_{args.task}")
    logger.log_config(vars(args))

    save_dir = f"checkpoints/hrm_phase1_{args.task}"
    trainer = Trainer(
        model, optimizer, scheduler,
        train_loader, val_loader,
        {"epochs": args.epochs, "grad_clip": 1.0,
         "save_every": 5, "save_dir": save_dir},
        logger,
    )
    trainer.train()

    final = trainer.eval()
    print(f"\n[hrm-phase1] Final val_acc={final['accuracy']:.4f}  "
          f"(base was {base_metrics['accuracy']:.4f}  "
          f"delta={final['accuracy'] - base_metrics['accuracy']:+.4f})")

    # ---- Gate check ----
    gates_nonzero = any(
        abs(m.gate.item()) > 1e-3
        for m in model.modules()
        if isinstance(m, HRMAdapter)
    )
    gate_passed = final["accuracy"] > 0.07 and gates_nonzero
    print(f"\nStep 3 gate ({'PASS' if gate_passed else 'FAIL'}): "
          f"val_acc > 0.07 and gates non-zero")

    # ---- Calibration pass: collect SSM hidden state snapshots ----
    print("\n[hrm-phase1] Running calibration pass to collect SSM snapshots...")
    model.eval()
    collector = StateCollector(model)
    collector.register()

    with torch.no_grad():
        n_calib = 0
        for input_ids, _, _ in train_loader:
            input_ids = input_ids.to(device)
            model(input_ids)
            n_calib += input_ids.shape[0]
            if n_calib >= args.calib_samples:
                break

    collector.remove()
    snapshots = collector.get_snapshots()

    print(f"[hrm-phase1] Collected snapshots from {len(snapshots)} layer(s):")
    for name, s in snapshots.items():
        print(f"  {name}: shape={tuple(s.shape)}")

    snap_path = os.path.join(save_dir, "snapshots.pt")
    torch.save({"snapshots": snapshots, "calib_samples": n_calib}, snap_path)
    print(f"[hrm-phase1] Snapshots saved to {snap_path}")

    # Report learned gate values
    print("\n[hrm-phase1] Learned gate values (should be non-zero):")
    for name, mod in model.named_modules():
        if isinstance(mod, HRMAdapter):
            print(f"  {name}.gate = {mod.gate.item():.6f}")

    if not gate_passed:
        sys.exit(1)


def parse_args():
    TASKS = ["copy", "ar", "delay_copy", "selective_copy", "parity", "cumsum_mod",
             "parity_last", "cumsum_last", "majority"]
    p = argparse.ArgumentParser()
    p.add_argument("--load",          type=str,   default=None,
                   help="Pre-trained base checkpoint (checkpoints/base/best.pt)")
    p.add_argument("--task",          choices=TASKS, default="ar")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--state_dim",     type=int,   default=32,
                   help="SSM state dimension before BT reduction")
    p.add_argument("--lr",            type=float, default=1e-3,
                   help="Higher LR than LoRA -- SSM params train from random init")
    p.add_argument("--batch",         type=int,   default=64)
    p.add_argument("--copy_len",      type=int,   default=16)
    p.add_argument("--n_train",       type=int,   default=5_000)
    p.add_argument("--n_val",         type=int,   default=500)
    p.add_argument("--calib_samples", type=int,   default=500,
                   help="Number of training sequences to use for calibration pass")
    p.add_argument("--gate_init",     type=float, default=0.0,
                   help="Initial value for SSM gate scalar (0=zero-residual; >0 unblocks A/B/C gradients)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--log_dir",       type=str,   default="logs")
    # AR-specific
    p.add_argument("--n_pairs",       type=int,   default=4)
    p.add_argument("--n_keys",        type=int,   default=16)
    p.add_argument("--n_values",      type=int,   default=15)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
