"""
HRM Phase 2: evaluate (and optionally fine-tune) the BT-reduced adapter on any task.

Scientific question: after Balanced Truncation compresses state_dim 32 -> d_hat,
how much accuracy is preserved without ANY additional training?

Then we ask: does brief fine-tuning (just the 2 gate scalars) recover further?

Loads the reduced checkpoint from reduce_adapter.py, applies the reduced matrices,
and runs evaluation + optional fine-tuning.

Usage:
  python scripts/train_hrm_phase2.py --load checkpoints/hrm_reduced_ar/model_reduced.pt --task ar
  python scripts/train_hrm_phase2.py --load checkpoints/hrm_reduced_parity/model_reduced.pt --task parity
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
from src.data.long_range_tasks import make_long_parity_loaders, make_dfa_loaders, \
    PARITY_VOCAB, DFA_VOCAB
from src.data.utils import seed_everything
from src.training.trainer import Trainer, make_cosine_scheduler
from src.training.metrics import parameter_count
from src.utils.logging import Logger

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


def load_task_data(args, seed_offset: int = 0):
    """Return (train_loader, val_loader, chance_acc) for the chosen task."""
    seed = args.seed + seed_offset
    task = args.task

    if task == "copy":
        loaders = make_copy_loaders(
            copy_len=16, vocab_size=16,
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
    elif task == "long_parity":
        loaders = make_long_parity_loaders(
            T=args.task_T, n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed,
        )
        chance = 0.5
    elif task == "dfa":
        # Infer n_states from checkpoint vocab_size: dfa_vocab_size(n) = n+4
        _vc = getattr(args, '_ckpt_vocab_size', None)
        n_states = (_vc - 4) if _vc and _vc > 4 else 4
        loaders = make_dfa_loaders(
            T=args.task_T, n_train=args.n_train, n_val=args.n_val,
            batch_size=args.batch, seed=seed, n_states=n_states,
        )
        chance = 1.0 / n_states
    else:
        raise ValueError(f"Unknown task: {task}")

    return loaders[0], loaders[1], chance


def main(args):
    if not args.load:
        print("ERROR: --load <checkpoint_path> required")
        sys.exit(1)

    seed_everything(args.seed + 999)
    device = get_device()
    print(f"[hrm-phase2] device={device}  task={args.task}")

    # Load reduced checkpoint metadata
    ckpt = torch.load(args.load, map_location=device, weights_only=False)
    hrm_cfg      = ckpt["hrm_config"]
    phase1_state = ckpt["model_state_dict"]
    meta         = ckpt["reduction_meta"]   # {layer_name: {d_hat, A_hat, B_hat, C_hat, ...}}

    # Resolve model config -- prefer embedded checkpoint values over defaults
    _mc        = ckpt.get("model_config", {})
    vocab_size = _mc.get("vocab_size", _VOCAB_SIZE)
    seq_len    = _mc.get("seq_len",    _SEQ_LEN)

    # Report reduction metadata
    print(f"[hrm-phase2] model: seq_len={seq_len}  vocab_size={vocab_size}")
    print("\n[hrm-phase2] BT reduction results:")
    for name, m in meta.items():
        print(f"  {name}: {m['d_orig']} -> {m['d_hat']}  "
              f"H-inf error bound={m['error_bound']:.4f}")

    # Rebuild model using resolved config
    model, cfg = build_model(vocab_size=vocab_size, seq_len=seq_len)
    model = model.to(device)
    hrm_cfg["hrm"]["input_dim"]  = cfg.d_model
    hrm_cfg["hrm"]["output_dim"] = cfg.d_model
    model = inject_hrm(model, hrm_cfg)
    model = model.to(device)

    # Load weights -- strict=False handles from-scratch checkpoints
    missing, unexpected = model.load_state_dict(phase1_state, strict=False)
    if missing:
        print(f"[hrm-phase2] WARNING missing keys: {missing}")

    # Apply BT-reduced matrices to each adapter
    for name, adapter in model.named_modules():
        if isinstance(adapter, HRMAdapter) and name in meta:
            m = meta[name]
            adapter.replace_ssm_matrices(
                m["A_hat"].to(device),
                m["B_hat"].to(device),
                m["C_hat"].to(device),
            )

    print("\n[hrm-phase2] Reduced model built.")
    for name, adapter in model.named_modules():
        if isinstance(adapter, HRMAdapter):
            d_hat = adapter._A_bar_reduced.shape[0]
            print(f"  {name}: state_dim={d_hat}  gate={adapter.gate.item():.6f}")

    # Expose checkpoint vocab_size so load_task_data can infer n_states for DFA
    args._ckpt_vocab_size = vocab_size

    # Data loaders (same split as phase1/LoRA)
    train_loader, val_loader, chance = load_task_data(args, seed_offset=999)

    # ---- Eval BEFORE any fine-tuning (pure BT compression quality) ----
    print("\n[hrm-phase2] Evaluating BT-reduced model WITHOUT fine-tuning...")
    _dummy_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    _eval = Trainer(
        model, _dummy_opt, None,
        train_loader, val_loader,
        {"epochs": 0, "grad_clip": 0.0, "save_every": 9999,
         "save_dir": "/tmp/_noop_hrm_p2"},
        Logger(args.log_dir, run_name=f"hrm_phase2_zero_shot_{args.task}"),
    )
    zero_shot = _eval.eval()
    print(f"[hrm-phase2] Zero-shot (BT only): acc={zero_shot['accuracy']:.4f}  "
          f"loss={zero_shot['loss']:.4f}")

    if not args.finetune:
        _print_comparison(zero_shot["accuracy"], None, meta, args.task)
        return

    # ---- Optional fine-tune: only gate parameters ----
    # The reduced matrices encode the essential dynamics; only gate scalars adapt.
    print(f"\n[hrm-phase2] Fine-tuning gate scalars only ({args.finetune_epochs} epochs)...")

    # Freeze everything; then unfreeze only gate params
    for p in model.parameters():
        p.requires_grad_(False)
    n_gate = 0
    for mod in model.modules():
        if isinstance(mod, HRMAdapter):
            mod.gate.requires_grad_(True)
            n_gate += 1
    print(f"  Trainable: {n_gate} gate scalar(s)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    total_steps  = args.finetune_epochs * len(train_loader)
    warmup_steps = min(20, total_steps // 10)
    scheduler    = make_cosine_scheduler(optimizer, warmup_steps, total_steps)

    logger = Logger(args.log_dir, run_name=f"hrm_phase2_ft_{args.task}")
    logger.log_config(vars(args))

    trainer = Trainer(
        model, optimizer, scheduler,
        train_loader, val_loader,
        {"epochs": args.finetune_epochs, "grad_clip": 1.0,
         "save_every": 5, "save_dir": f"checkpoints/hrm_phase2_{args.task}"},
        logger,
    )
    trainer.train()

    ft_metrics = trainer.eval()
    _print_comparison(zero_shot["accuracy"], ft_metrics["accuracy"], meta, args.task)


def _print_comparison(zero_shot_acc, ft_acc, meta, task="ar"):
    d_hat_vals = [m["d_hat"] for m in meta.values()]
    d_hats_str = "/".join(str(d) for d in d_hat_vals)
    d_orig = list(meta.values())[0]["d_orig"]

    print("\n" + "=" * 60)
    print(f"Step 5 Results -- BT Compression Quality  [{task}]")
    print("=" * 60)
    print(f"  HRM-BT (d={d_hats_str}, zero-shot):       {zero_shot_acc:.4f}")
    if ft_acc is not None:
        print(f"  HRM-BT (d={d_hats_str}, gate fine-tune):  {ft_acc:.4f}")
    print(f"  Compression ratio: {d_orig} -> {d_hats_str} states")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load",           type=str,   default=None,
                   help="Reduced checkpoint (model_reduced.pt)")
    p.add_argument("--task",           type=str,   default="ar",
                   help="Task name; use 'long_parity' or 'dfa' for sweep tasks")
    p.add_argument("--task_T",         type=int,   default=None,
                   help="T for long_parity/dfa tasks (number of bits / DFA input length)")
    p.add_argument("--finetune",       action="store_true",
                   help="Run brief gate fine-tuning after zero-shot eval")
    p.add_argument("--finetune_epochs",type=int,   default=5)
    p.add_argument("--lr",             type=float, default=3e-3)
    p.add_argument("--batch",          type=int,   default=64)
    p.add_argument("--n_train",        type=int,   default=5_000)
    p.add_argument("--n_val",          type=int,   default=500)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--log_dir",        type=str,   default="logs")
    p.add_argument("--n_pairs",        type=int,   default=4)
    p.add_argument("--n_keys",         type=int,   default=16)
    p.add_argument("--n_values",       type=int,   default=15)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
