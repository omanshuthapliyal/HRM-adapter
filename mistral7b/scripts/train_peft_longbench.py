#!/usr/bin/env python3
"""
Unified PEFT fine-tuning script for LongBench tasks.

Adapter types: hrm | lora | dora | adalora | qlora

Data sources (HuggingFace Hub):
  narrativeqa  ->  deepmind/narrativeqa  (train split, uses document summaries)
  quality      ->  emozilla/quality       (train split)
  qmsum        ->  pszemraj/qmsum-cleaned (train split)

Usage:
  # Local GPU validation (n_train=200 subset, GPT-2 medium):
  python experiments/cluster/scripts/train_peft_longbench.py \\
    --config experiments/local_gpu/configs/hrm_gpt2medium.yaml \\
    --task narrativeqa --seed 42 --n_train 200 \\
    --output_dir experiments/local_gpu/logs/hrm_narrativeqa_s42

  # Cluster full run (Mistral-7B):
  python experiments/cluster/scripts/train_peft_longbench.py \\
    --config experiments/cluster/configs/hrm_mistral7b.yaml \\
    --task narrativeqa --seed 42 \\
    --output_dir experiments/cluster/checkpoints/hrm_narrativeqa_s42
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Ensure repo root is on PYTHONPATH
_REPO_ROOT = Path(__file__).resolve().parents[1]  # experiments/cluster/
sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.hf_injection import inject_hrm_hf  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--task", required=True,
                   choices=["narrativeqa", "quality", "qmsum"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_train", type=int, default=None,
                   help="Limit training examples (for quick validation runs)")
    p.add_argument("--max_input_length", type=int, default=None,
                   help="Override task.max_input_length from config")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--frozen_gate", action="store_true",
                   help="[Ablation B2] Freeze HRM gate at gate_init value (gate.requires_grad=False). "
                        "Tests whether SSM contribution at inference is required for gains, or "
                        "whether training-time inductive bias alone explains HRM's advantage.")
    p.add_argument("--gate_init_override", type=float, default=None,
                   help="Override adapter.gate_init from config (useful with --frozen_gate).")
    p.add_argument("--bt_reduce_init", type=str, default=None,
                   help="Path to .pt file from reduce_mistral_hrm.py. "
                        "Overrides state_dim from config and initializes each layer's "
                        "SSM matrices (A, B, C) from BT-reduced matrices. "
                        "state_dim is set to the max d_hat across all layers.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset loading and formatting
# ---------------------------------------------------------------------------

def _quality_answer_idx(ex) -> int:
    """Return 0-indexed answer position, tolerating different HF field names/indexing."""
    # emozilla/quality field name varies across dataset versions:
    #   gold_label (1-indexed int), answer (0-indexed int or letter A-D), label (1-indexed int)
    for field, one_indexed in (("gold_label", True), ("label", True), ("answer", False)):
        if field in ex:
            val = ex[field]
            if isinstance(val, str):
                # Stored as letter "A"/"B"/"C"/"D"
                if val.strip().upper() in "ABCD":
                    return ord(val.strip().upper()) - ord("A")
                val = int(val)
            return int(val) - (1 if one_indexed else 0)
    raise KeyError(
        f"Cannot find answer field in quality example. "
        f"Available keys: {sorted(ex.keys())}"
    )


TASK_CONFIGS = {
    "narrativeqa": {
        "hf_path": "deepmind/narrativeqa",
        "split": "train",
        "prompt_fn": lambda ex: (
            f"Document:\n{ex['document']['text']}\n\n"
            f"Question: {ex['question']['text']}\nAnswer:"
        ),
        "answer_fn": lambda ex: ex["answers"][0]["text"],
    },
    "quality": {
        "hf_path": "emozilla/quality",
        "split": "train",
        "prompt_fn": lambda ex: (
            f"Article:\n{ex['article']}\n\n"
            f"Question: {ex['question']}\n"
            f"A: {ex['options'][0]}\nB: {ex['options'][1]}\n"
            f"C: {ex['options'][2]}\nD: {ex['options'][3]}\nAnswer:"
        ),
        "answer_fn": lambda ex: ["A", "B", "C", "D"][_quality_answer_idx(ex)],
    },
    "qmsum": {
        "hf_path": "pszemraj/qmsum-cleaned",
        "split": "train",
        "prompt_fn": lambda ex: (
            f"Meeting transcript:\n{ex['input']}\n\nSummary:"
        ),
        "answer_fn": lambda ex: ex["output"],
    },
}


def build_dataset(task: str, tokenizer, max_input_length: int,
                  max_new_tokens: int, n_train=None):
    cfg = TASK_CONFIGS[task]
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    if n_train is not None:
        ds = ds.select(range(min(n_train, len(ds))))

    def _middle_truncate_prompt(prompt: str, budget: int) -> list:
        """Tokenize prompt, middle-truncating the document to fit within budget tokens.
        Always preserves the task suffix (question/instruction) after the last \\n\\n."""
        ids = tokenizer(prompt, add_special_tokens=True).input_ids
        if len(ids) <= budget:
            return ids
        split_idx = prompt.rfind("\n\n")
        if split_idx == -1:
            return ids[:budget]
        doc_part = prompt[:split_idx]
        suffix = prompt[split_idx:]
        suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
        doc_budget = budget - len(suffix_ids) - 1
        doc_ids = tokenizer(doc_part, add_special_tokens=False).input_ids
        if len(doc_ids) > doc_budget:
            half = doc_budget // 2
            doc_ids = doc_ids[:half] + doc_ids[len(doc_ids) - (doc_budget - half):]
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        return bos + doc_ids + suffix_ids

    def tokenize(ex):
        prompt = cfg["prompt_fn"](ex)
        answer = cfg["answer_fn"](ex)

        prompt_ids = _middle_truncate_prompt(prompt, max_input_length)
        answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
        full_ids = (prompt_ids + answer_ids)[: max_input_length + max_new_tokens]

        n_prompt = len(prompt_ids)
        labels = [-100] * n_prompt + full_ids[n_prompt:]
        labels = labels[: len(full_ids)]

        enc = {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}
        return enc

    ds = ds.map(tokenize, remove_columns=ds.column_names,
                desc="Tokenizing", load_from_cache_file=False)
    return ds


# ---------------------------------------------------------------------------
# HRM state helpers (used by HRMChunkedTrainer)
# ---------------------------------------------------------------------------

def _get_hrm_adapters(model):
    from src.adapters.hrm_adapter import HRMAdapter
    return [m for m in model.modules() if isinstance(m, HRMAdapter)]


def _reset_hrm_states(model, batch_size, device=None):
    for a in _get_hrm_adapters(model):
        d = next(a.parameters()).device if device is None else device
        a.reset_state(batch_size, d)


def _clear_hrm_states(model):
    for a in _get_hrm_adapters(model):
        a.clear_state()


def _clip_hrm_states(model, max_norm: float = 10.0):
    """Clip SSM hidden state norm to prevent explosion during prior no_grad chunks."""
    for a in _get_hrm_adapters(model):
        if a.hidden_state is not None:
            norm = a.hidden_state.norm(dim=-1, keepdim=True)
            a.hidden_state = a.hidden_state * (max_norm / norm.clamp(min=max_norm))


# ---------------------------------------------------------------------------
# Chunked dataset for HRM + NarrativeQA (end-truncate doc, keep tail)
# ---------------------------------------------------------------------------

def build_chunked_dataset(task: str, tokenizer, max_input_length: int,
                          max_new_tokens: int, n_chunks: int = 4, n_train=None):
    cfg = TASK_CONFIGS[task]
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    if n_train is not None:
        ds = ds.select(range(min(n_train, len(ds))))
    total_budget = n_chunks * max_input_length

    def tokenize(ex):
        prompt = cfg["prompt_fn"](ex)
        answer = cfg["answer_fn"](ex)
        split_idx = prompt.rfind("\n\n")
        if split_idx == -1:
            split_idx = len(prompt)
        doc_part = prompt[:split_idx]
        suffix = prompt[split_idx:]
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
        answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
        overhead = len(bos) + len(suffix_ids) + len(answer_ids)
        doc_budget = total_budget - overhead
        doc_ids = tokenizer(doc_part, add_special_tokens=False).input_ids
        if len(doc_ids) > doc_budget:
            doc_ids = doc_ids[-doc_budget:]  # keep tail -- end of doc closest to question
        full_ids = (bos + doc_ids + suffix_ids + answer_ids)[:total_budget]
        n_prompt = len(bos) + len(doc_ids) + len(suffix_ids)
        labels = ([-100] * n_prompt + answer_ids)[:len(full_ids)]
        attn = [1] * len(full_ids)
        return {"input_ids": full_ids, "attention_mask": attn, "labels": labels}

    ds = ds.map(tokenize, remove_columns=ds.column_names,
                desc="Tokenizing (chunked)", load_from_cache_file=False)
    return ds


# ---------------------------------------------------------------------------
# Chunked trainer -- truncated BPTT for HRM NarrativeQA
# ---------------------------------------------------------------------------

class HRMChunkedTrainer(Trainer):
    def __init__(self, *args, chunk_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        labels    = inputs["labels"]
        attn_mask = inputs["attention_mask"]
        T = input_ids.shape[1]

        _reset_hrm_states(model, batch_size=input_ids.shape[0])

        n_chunks = math.ceil(T / self.chunk_size)
        for i in range(n_chunks - 1):
            s = i * self.chunk_size
            e = min((i + 1) * self.chunk_size, T)
            with torch.no_grad():
                model(input_ids=input_ids[:, s:e],
                      attention_mask=attn_mask[:, s:e],
                      use_cache=False)
            _clip_hrm_states(model)  # prevent state explosion before gradient chunk
            # _forward_stateful does h.detach() -- truncated BPTT is automatic

        s = (n_chunks - 1) * self.chunk_size
        outputs = model(
            input_ids=input_ids[:, s:],
            attention_mask=attn_mask[:, s:],
            labels=labels[:, s:],
            use_cache=False,
        )
        _clear_hrm_states(model)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_apply_adapter(cfg: dict):
    model_cfg = cfg["model"]
    adapter_cfg = cfg["adapter"]
    model_name = model_cfg["name"]
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "float16"), torch.float16)
    adapter_type = adapter_cfg["type"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_type == "hrm":
        # With 2 GPUs: cap each GPU to 11GiB so "auto" is forced to split the 14GB model.
        # Without max_memory, "auto" fits the whole model on GPU 0 and OOMs on activations.
        n_gpu = torch.cuda.device_count()
        max_memory = {i: "11GiB" for i in range(n_gpu)} if n_gpu > 1 else None
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            max_memory=max_memory,
        )
        inject_hrm_hf(
            model,
            state_dim=adapter_cfg["state_dim"],
            gate_init=adapter_cfg.get("gate_init", 0.1),
            dt_init=adapter_cfg.get("dt_init", 0.01),
            dt_min=adapter_cfg.get("dt_min", 1e-4),
            dt_max=adapter_cfg.get("dt_max", 0.1),
        )
        # Align injected HRM adapters to their block's device (needed for multi-GPU)
        for block in model.model.layers:
            if hasattr(block, "hrm"):
                block.hrm.to(next(block.mlp.parameters()).device)

    elif adapter_type in ("lora", "dora"):
        from peft import LoraConfig, get_peft_model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map={"": 0},
        )
        peft_cfg = LoraConfig(
            r=adapter_cfg["lora_rank"],
            lora_alpha=adapter_cfg["lora_alpha"],
            lora_dropout=adapter_cfg.get("lora_dropout", 0.05),
            target_modules=adapter_cfg["target_modules"],
            use_dora=(adapter_type == "dora"),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    elif adapter_type == "adalora":
        from peft import AdaLoraConfig, get_peft_model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map={"": 0},
        )
        _tfinal = adapter_cfg.get("tfinal", 1000)
        peft_cfg = AdaLoraConfig(
            target_r=adapter_cfg["target_r"],
            init_r=adapter_cfg.get("init_r", adapter_cfg["target_r"] + 8),
            lora_alpha=adapter_cfg["lora_alpha"],
            lora_dropout=adapter_cfg.get("lora_dropout", 0.05),
            target_modules=adapter_cfg["target_modules"],
            tinit=adapter_cfg.get("tinit", 200),
            tfinal=_tfinal,
            deltaT=adapter_cfg.get("deltaT", 10),
            beta1=adapter_cfg.get("beta1", 0.85),
            beta2=adapter_cfg.get("beta2", 0.85),
            orth_reg_weight=adapter_cfg.get("orth_reg_weight", 0.5),
            total_step=adapter_cfg.get("total_step", _tfinal * 2),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    elif adapter_type == "qlora":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=dtype_map.get(
                model_cfg.get("bnb_4bit_compute_dtype", "float16"), torch.float16
            ),
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model)
        peft_cfg = LoraConfig(
            r=adapter_cfg["lora_rank"],
            lora_alpha=adapter_cfg["lora_alpha"],
            lora_dropout=adapter_cfg.get("lora_dropout", 0.05),
            target_modules=adapter_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_tag = "_frozen_gate" if args.frozen_gate else ""
    print(f"\n{'='*60}")
    print(f"  Task: {args.task}  |  Adapter: {cfg['adapter']['type']}{frozen_tag}")
    print(f"  Model: {cfg['model']['name']}  |  Seed: {args.seed}")
    if args.n_train:
        print(f"  n_train: {args.n_train} (validation subset)")
    print(f"{'='*60}\n")

    # Apply CLI gate_init override before building the model
    if args.gate_init_override is not None:
        cfg["adapter"]["gate_init"] = args.gate_init_override

    # BT-reduce init: override state_dim to max d_hat before inject
    bt_data = None
    if args.bt_reduce_init is not None:
        bt_data = torch.load(args.bt_reduce_init, map_location="cpu")
        valid_d = [d for d in bt_data["d_hat"] if d > 0]
        bt_state_dim = max(valid_d)
        print(f"[bt_reduce_init] Loaded {args.bt_reduce_init}")
        print(f"[bt_reduce_init] d_hat: min={min(valid_d)} median={sorted(valid_d)[len(valid_d)//2]} "
              f"max={bt_state_dim}  (eps={bt_data['meta']['eps']})")
        print(f"[bt_reduce_init] Overriding state_dim: {cfg['adapter']['state_dim']} -> {bt_state_dim}")
        cfg["adapter"]["state_dim"] = bt_state_dim

    # --- Model + adapter ---
    model, tokenizer = load_model_and_apply_adapter(cfg)

    # BT matrix initialization: copy reduced A_hat, B_hat, C_hat into injected HRM layers.
    # All SSM parameters must stay at bt_state_dim (max d_hat across layers) so that
    # log_A, log_dt, B, C all have the same leading dimension -- required by _get_discrete_matrices.
    # Layers with d_hat_i < bt_state_dim get zero-padded B and C (dead modes); log_A keeps its
    # random init for the padding dims so they stay stable but contribute nothing via B/C=0.
    if bt_data is not None and cfg["adapter"]["type"] == "hrm":
        n_init = 0
        for layer_idx, block in enumerate(model.model.layers):
            if not hasattr(block, "hrm"):
                continue
            layer_bt = bt_data["layers"][layer_idx] if layer_idx < len(bt_data["layers"]) else None
            if layer_bt is None:
                continue
            d_hat = bt_data["d_hat"][layer_idx]
            ssm = block.hrm.ssm
            full_dim = ssm.log_A.shape[0]   # bt_state_dim, already allocated correctly
            with torch.no_grad():
                A_hat = layer_bt["A_hat"].to(ssm.log_A.device)
                B_hat = layer_bt["B_hat"].to(ssm.B.device)
                C_hat = layer_bt["C_hat"].to(ssm.C.device)
                # log_A: set first d_hat dims from BT diagonal; leave padding dims at random init
                a_diag = A_hat.diag().clamp(1e-6, 1.0 - 1e-6)
                ssm.log_A.data[:d_hat].copy_((-a_diag.log()).to(ssm.log_A.dtype))
                # B: BT values for first d_hat rows, zero for padding (dead modes)
                ssm.B.data.zero_()
                ssm.B.data[:d_hat].copy_(B_hat.to(ssm.B.dtype))
                # C: BT values for first d_hat cols, zero for padding
                ssm.C.data.zero_()
                ssm.C.data[:, :d_hat].copy_(C_hat.to(ssm.C.dtype))
                # log_dt: leave at its initialized values (size full_dim) -- no change needed
            n_init += 1
        print(f"[bt_reduce_init] Initialized SSM matrices in {n_init} HRM layers from BT "
              f"(d_hat per layer padded to full_dim={bt_state_dim})")

    # Frozen gate ablation (B2): freeze gate after injection so alpha is fixed at gate_init.
    # This tests whether inference-time SSM contribution is required for HRM's gains,
    # or whether training-time sequential inductive bias alone explains the advantage.
    if args.frozen_gate and cfg["adapter"]["type"] == "hrm":
        n_frozen = 0
        for block in model.model.layers:
            if hasattr(block, "hrm"):
                block.hrm.gate.requires_grad_(False)
                n_frozen += 1
        print(f"[frozen_gate] Froze gate in {n_frozen} HRM layers "
              f"(alpha={cfg['adapter'].get('gate_init', 0.1):.3f}, fixed throughout training)")

    trainable_params = count_trainable(model)
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Dataset ---
    task_cfg = cfg["task"]
    adapter_type = cfg["adapter"]["type"]
    max_input_length = args.max_input_length or task_cfg["max_input_length"]
    use_chunked = (adapter_type == "hrm" and args.task == "narrativeqa")
    if use_chunked:
        print("  [chunked BPTT mode: 4 chunks x max_input_length]")
        dataset = build_chunked_dataset(
            args.task, tokenizer,
            max_input_length=max_input_length,
            max_new_tokens=task_cfg["max_new_tokens"],
            n_chunks=4,
            n_train=args.n_train,
        )
    else:
        dataset = build_dataset(
            args.task, tokenizer,
            max_input_length=max_input_length,
            max_new_tokens=task_cfg["max_new_tokens"],
            n_train=args.n_train,
        )
    # 90/10 train/val split
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # --- Training args ---
    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        learning_rate=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        max_grad_norm=train_cfg.get("grad_clip", 1.0),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=False if use_chunked else train_cfg.get("gradient_checkpointing", False),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=cfg["output"].get("log_every_n_steps", 50),
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )
    # Exclude gate parameters from weight decay -- weight decay on a scalar gate
    # pulls it toward zero regardless of task signal, causing SSM collapse.
    if adapter_type == "hrm":
        no_decay = ["bias", "layer_norm", "layernorm", "gate"]
        grouped_params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay) and p.requires_grad], "weight_decay": training_args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(grouped_params, lr=training_args.learning_rate)
        trainer_kwargs["optimizers"] = (optimizer, None)

    if use_chunked:
        trainer = HRMChunkedTrainer(**trainer_kwargs, chunk_size=max_input_length)
    else:
        trainer = Trainer(**trainer_kwargs)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Trainer with load_best_model_at_end=True only loads the best checkpoint
    # into memory; it does NOT write it back to output_dir automatically.
    trainer.save_model(str(output_dir))

    # --- Save metadata ---
    meta = {
        "task": args.task,
        "adapter": cfg["adapter"]["type"],
        "model": cfg["model"]["name"],
        "seed": args.seed,
        "trainable_params": trainable_params,
        "n_train": len(train_ds),
        "elapsed_s": round(elapsed, 1),
    }
    with open(output_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved to {output_dir}  |  elapsed: {elapsed/60:.1f} min")
    print(f"RESULT task={args.task} adapter={cfg['adapter']['type']} "
          f"seed={args.seed} trainable_params={trainable_params}")


if __name__ == "__main__":
    main()
