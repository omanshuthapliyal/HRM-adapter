#!/usr/bin/env python3
"""
LongBench evaluation for fine-tuned PEFT checkpoints.

Uses the official THUDM/LongBench dataset (HuggingFace) and evaluation metrics.
Supported tasks: narrativeqa, quality, qmsum

Metrics:
  narrativeqa  ->  F1 (token-level, case-insensitive)
  quality      ->  Accuracy
  qmsum        ->  ROUGE-1 / ROUGE-2 / ROUGE-L

Usage:
  python experiments/cluster/scripts/eval_longbench.py \\
    --checkpoint experiments/cluster/checkpoints/hrm_narrativeqa_s42 \\
    --config experiments/cluster/configs/hrm_mistral7b.yaml \\
    --task narrativeqa \\
    --output_dir experiments/cluster/logs/eval_hrm_narrativeqa_s42

  # For HRM, also pass the original adapter type so the model is rebuilt correctly:
  python eval_longbench.py --checkpoint ... --config ... --task narrativeqa
"""

import argparse
import csv
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

_CLUSTER_ROOT = Path(__file__).resolve().parents[1]  # experiments/cluster/
_SCRIPTS_DIR = Path(__file__).resolve().parent       # experiments/cluster/scripts/
sys.path.insert(0, str(_CLUSTER_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

from train_peft_longbench import (  # noqa: E402
    load_model_and_apply_adapter,
    _quality_answer_idx,
)


# ---------------------------------------------------------------------------
# Metric implementations (mirrors LongBench official scorer)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def rouge_l(pred: str, ref: str) -> float:
    pred_tokens = _normalize(pred).split()
    ref_tokens = _normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def rouge_n(pred: str, ref: str, n: int) -> float:
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    pred_tokens = _normalize(pred).split()
    ref_tokens = _normalize(ref).split()
    pred_ng = ngrams(pred_tokens, n)
    ref_ng = ngrams(ref_tokens, n)
    overlap = sum((pred_ng & ref_ng).values())
    ref_total = sum(ref_ng.values())
    if ref_total == 0:
        return 0.0
    return overlap / ref_total


TASK_METRIC_FNS = {
    "narrativeqa": lambda p, g: {"f1": f1_score(p, g)},
    "quality": lambda p, g: {"accuracy": exact_match(p.strip()[:1], g.strip()[:1])},
    "qmsum": lambda p, g: {
        "rouge1": rouge_n(p, g, 1),
        "rouge2": rouge_n(p, g, 2),
        "rougeL": rouge_l(p, g),
    },
}

# LongBench HuggingFace dataset config names
LONGBENCH_TASK_MAP = {
    "narrativeqa": "narrativeqa",
    "quality": "quality",
    "qmsum": "qmsum",
}

# How to extract prompt + gold from source datasets (not THUDM/LongBench)
LONGBENCH_FORMAT = {
    # deepmind/narrativeqa test split: document.text, question.text, answers (list of dicts with 'text')
    "narrativeqa": {
        "prompt_fn": lambda ex: (
            f"Document:\n{ex['document']['text']}\n\n"
            f"Question: {ex['question']['text']}\nAnswer:"
        ),
        "gold_fn": lambda ex: [a["text"] for a in ex["answers"]],
    },
    # emozilla/quality (held-out 10%) -- article, question, options, answer field (varies by version)
    "quality": {
        "prompt_fn": lambda ex: (
            f"Article:\n{ex['article']}\n\n"
            f"Question: {ex['question']}\n"
            f"A: {ex['options'][0]}\nB: {ex['options'][1]}\n"
            f"C: {ex['options'][2]}\nD: {ex['options'][3]}\nAnswer:"
        ),
        "gold_fn": lambda ex: [["A", "B", "C", "D"][_quality_answer_idx(ex)]],
    },
    # pszemraj/qmsum-cleaned (held-out 10%) -- input, output
    "qmsum": {
        "prompt_fn": lambda ex: (
            f"Meeting transcript:\n{ex['input']}\n\nSummary:"
        ),
        "gold_fn": lambda ex: [ex["output"]],
    },
}


# ---------------------------------------------------------------------------
# HRM chunked-eval helpers
# ---------------------------------------------------------------------------

def _get_hrm_adapters(model):
    """Collect all HRMAdapter instances injected into the model."""
    from src.adapters.hrm_adapter import HRMAdapter
    return [m for m in model.modules() if isinstance(m, HRMAdapter)]


def _reset_hrm_states(model, batch_size: int, device=None):
    for a in _get_hrm_adapters(model):
        d = next(a.parameters()).device if device is None else device
        a.reset_state(batch_size, d)


def _clear_hrm_states(model):
    for a in _get_hrm_adapters(model):
        a.clear_state()


def _chunked_generate(model, tokenizer, prompt: str, max_input: int,
                      max_new_tokens: int, device, adapter_type: str,
                      pad_token_id: int) -> str:
    """
    Chunked generation for apples-to-apples long-context comparison.

    ALL methods generate from the IDENTICAL final chunk (last max_input tokens
    of the document + task suffix). HRM additionally pre-processes all prior
    chunks to accumulate SSM state -- its only advantage.

    Chunk size = max_input - suffix_tokens - 1 (BOS), so the final generation
    input is always exactly max_input tokens for every method.
    """
    split_idx = prompt.rfind("\n\n")
    if split_idx == -1:
        # No doc/suffix split -- fall back to standard single-chunk path
        inputs = _make_input(tokenizer, prompt, max_input)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, repetition_penalty=1.3,
                                 pad_token_id=pad_token_id, use_cache=True)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    doc_part = prompt[:split_idx]
    suffix   = prompt[split_idx:]   # "\n\nQuestion: ...\nAnswer:" or "\n\nSummary:"

    bos        = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    doc_ids    = tokenizer(doc_part, add_special_tokens=False).input_ids
    chunk_size = max_input - len(suffix_ids) - len(bos)  # doc tokens per chunk

    if chunk_size <= 0:
        # Suffix alone fills the budget -- just generate from suffix
        ids = bos + suffix_ids
        t   = torch.tensor([ids]).to(device)
        with torch.no_grad():
            out = model.generate(input_ids=t, attention_mask=torch.ones_like(t),
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.3, pad_token_id=pad_token_id,
                                 use_cache=True)
        gen = out[0][t.shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    # Cap document to last MAX_PRIOR_CHUNKS+1 chunks (keeps tail of document where
    # answers typically reside; bounds HRM eval to ~5 forward passes per example).
    MAX_PRIOR_CHUNKS = 0  # 0 = stateless eval: matches stateless training (no OOD state)
    if len(doc_ids) > (MAX_PRIOR_CHUNKS + 1) * chunk_size:
        doc_ids = doc_ids[-(MAX_PRIOR_CHUNKS + 1) * chunk_size:]

    # Split document into equal-sized chunks
    chunks = [doc_ids[i:i + chunk_size] for i in range(0, max(1, len(doc_ids)), chunk_size)]

    # --- HRM: reset state so _forward_stateful runs during prefill + generation ---
    # Must always reset (not just when len(chunks)>1): without reset, hidden_state=None
    # triggers _forward_fft on single tokens during generation (= CxBxx, no recurrence).
    # With reset to h=0, _forward_stateful accumulates state over the full prompt during
    # model.generate's prefill pass, then continues token-by-token -- matching training.
    if adapter_type == "hrm":
        _reset_hrm_states(model, batch_size=1)
        for chunk in chunks[:-1]:
            chunk_t = torch.tensor([chunk]).to(device)
            with torch.no_grad():
                model(input_ids=chunk_t,
                      attention_mask=torch.ones_like(chunk_t))

    # --- Final generation input: last chunk + suffix (IDENTICAL for all methods) ---
    final_ids = bos + chunks[-1] + suffix_ids
    t = torch.tensor([final_ids]).to(device)
    with torch.no_grad():
        out = model.generate(input_ids=t, attention_mask=torch.ones_like(t),
                             max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=1.3, pad_token_id=pad_token_id,
                             use_cache=True)

    if adapter_type == "hrm":
        _clear_hrm_states(model)

    gen = out[0][t.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def _make_input(tokenizer, prompt: str, max_length: int) -> dict:
    """
    Tokenize prompt, preserving the task suffix (question/instruction) by
    middle-truncating the document if the full prompt exceeds max_length.

    Splits on the last double-newline to separate document from task suffix.
    """
    ids = tokenizer(prompt, add_special_tokens=True).input_ids
    if len(ids) <= max_length:
        return tokenizer(prompt, return_tensors="pt",
                         truncation=True, max_length=max_length)

    split_idx = prompt.rfind("\n\n")
    if split_idx == -1:
        return tokenizer(prompt, return_tensors="pt",
                         truncation=True, max_length=max_length)

    doc_part = prompt[:split_idx]
    suffix = prompt[split_idx:]  # "\n\nQuestion: ...\nAnswer:" or "\n\nSummary:"

    suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    doc_budget = max_length - len(suffix_ids) - 1  # -1 for BOS token

    doc_ids = tokenizer(doc_part, add_special_tokens=False).input_ids
    if len(doc_ids) > doc_budget:
        half = doc_budget // 2
        doc_ids = doc_ids[:half] + doc_ids[len(doc_ids) - (doc_budget - half):]

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    combined = bos + doc_ids + suffix_ids
    t = torch.tensor([combined])
    return {"input_ids": t, "attention_mask": torch.ones_like(t)}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Directory saved by train_peft_longbench.py")
    p.add_argument("--config", required=True, help="YAML config used during training")
    p.add_argument("--task", required=True,
                   choices=["narrativeqa", "quality", "qmsum"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Override max_new_tokens from config")
    p.add_argument("--max_input_length", type=int, default=None,
                   help="Override task.max_input_length from config")
    p.add_argument("--max_examples", type=int, default=None,
                   help="Limit evaluation examples (for debugging)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Eval: {args.task}  |  Adapter: {cfg['adapter']['type']}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    # --- Auto-detect state_dim from checkpoint (handles BT-reduced checkpoints) ---
    checkpoint_path = Path(args.checkpoint)
    if cfg["adapter"]["type"] == "hrm":
        _probe_path = checkpoint_path / "model.safetensors"
        _probe_shards = sorted(checkpoint_path.glob("model-*.safetensors"))
        _probe_file = _probe_path if _probe_path.exists() else (_probe_shards[0] if _probe_shards else None)
        if _probe_file is not None:
            from safetensors.torch import load_file as _lf
            _probe = _lf(str(_probe_file), device="cpu")
            _key = "model.layers.0.hrm.ssm.log_A"
            if _key in _probe:
                _detected = _probe[_key].shape[0]
                if _detected != cfg["adapter"]["state_dim"]:
                    print(f"[eval] BT checkpoint detected: state_dim {cfg['adapter']['state_dim']} -> {_detected}")
                    cfg["adapter"]["state_dim"] = _detected
            del _probe

    # --- Load model (rebuild architecture, then load weights) ---
    model, tokenizer = load_model_and_apply_adapter(cfg)

    checkpoint_path = Path(args.checkpoint)
    # HF Trainer saves checkpoint in epoch subdirectories; find the best model
    # (train_peft_longbench uses load_best_model_at_end=True so the base dir has it)
    safetensors_path = checkpoint_path / "model.safetensors"
    bin_path = checkpoint_path / "pytorch_model.bin"
    shards = sorted(checkpoint_path.glob("model-*.safetensors"))
    # PEFT saves adapter_model.safetensors / adapter_model.bin + adapter_config.json
    adapter_safetensors = checkpoint_path / "adapter_model.safetensors"
    adapter_bin = checkpoint_path / "adapter_model.bin"
    adapter_config = checkpoint_path / "adapter_config.json"

    if cfg["adapter"]["type"] == "hrm":
        if safetensors_path.exists():
            from safetensors.torch import load_file
            state = load_file(str(safetensors_path), device="cpu")
        elif shards:
            from safetensors.torch import load_file
            state = {}
            for shard in shards:
                state.update(load_file(str(shard), device="cpu"))
        elif bin_path.exists():
            state = torch.load(bin_path, map_location="cpu")
        else:
            state = {}
            print(f"WARNING: no weights found at {checkpoint_path} -- evaluating untrained HRM.")
        hrm_state = {k: v for k, v in state.items() if "hrm" in k}
        model.load_state_dict(hrm_state, strict=False)
        print(f"Loaded {len(hrm_state)} HRM parameter tensors from checkpoint.")
    elif (adapter_config.exists() or adapter_safetensors.exists() or adapter_bin.exists()
          or safetensors_path.exists() or bin_path.exists() or shards):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
        print("Loaded PEFT adapter from checkpoint.")
    else:
        print(f"WARNING: no weights found at {checkpoint_path} -- evaluating untrained model.")

    model.eval()
    device = next(p for p in model.parameters() if p.requires_grad).device \
             if any(p.requires_grad for p in model.parameters()) \
             else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load eval data ---
    # THUDM/LongBench uses a custom dataset script that is blocked in datasets>=2.18.
    # Instead we use held-out splits of the same source datasets used for training,
    # ensuring a clean train/eval split with no data leakage.
    if args.task == "narrativeqa":
        # deepmind/narrativeqa has an official test split -- use it directly.
        lb_dataset = load_dataset("deepmind/narrativeqa", split="test")
    elif args.task == "quality":
        # emozilla/quality: hold out 10% (same seed as training split).
        full_ds = load_dataset("emozilla/quality", split="train")
        lb_dataset = full_ds.train_test_split(test_size=0.1, seed=42)["test"]
    elif args.task == "qmsum":
        # pszemraj/qmsum-cleaned: hold out 10% (same seed as training split).
        full_ds = load_dataset("pszemraj/qmsum-cleaned", split="train")
        lb_dataset = full_ds.train_test_split(test_size=0.1, seed=42)["test"]
    else:
        raise ValueError(f"Unknown task: {args.task}")
    if args.max_examples:
        lb_dataset = lb_dataset.select(range(min(args.max_examples, len(lb_dataset))))

    fmt = LONGBENCH_FORMAT[args.task]
    metric_fn = TASK_METRIC_FNS[args.task]
    max_new_tokens = args.max_new_tokens or cfg["task"].get("max_new_tokens", 128)
    max_input = args.max_input_length or cfg["task"]["max_input_length"]

    # --- Generation loop ---
    adapter_type = cfg["adapter"]["type"]
    results = []
    all_scores = {}

    for ex in tqdm(lb_dataset, desc=f"Evaluating {args.task}"):
        prompt = fmt["prompt_fn"](ex)
        golds = fmt["gold_fn"](ex)
        if isinstance(golds, str):
            golds = [golds]

        if adapter_type == "hrm":
            # HRM uses chunked generation: prior document chunks accumulate SSM
            # state before generation; all methods generate from the same final
            # chunk so the generation context is identical for a fair comparison.
            pred = _chunked_generate(
                model, tokenizer, prompt, max_input, max_new_tokens,
                device, adapter_type, pad_token_id=tokenizer.eos_token_id,
            )
        else:
            inputs = _make_input(tokenizer, prompt, max_input)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Score against all gold answers; take max
        scores = {k: max(metric_fn(pred, g)[k] for g in golds)
                  for k in metric_fn(pred, golds[0])}

        for k, v in scores.items():
            all_scores.setdefault(k, []).append(v)

        results.append({"prompt": prompt[:200], "pred": pred,
                        "gold": golds[0], **scores})

    # --- Aggregate ---
    agg = {k: sum(v) / len(v) for k, v in all_scores.items()}
    print(f"\n{'='*60}")
    print(f"  Results -- {args.task}  (n={len(results)})")
    for k, v in agg.items():
        print(f"    {k}: {v:.4f}")
    print(f"{'='*60}\n")

    # --- Save ---
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    # Append one row to a shared CSV for easy comparison
    csv_path = output_dir.parent / "longbench_results.csv"
    write_header = not csv_path.exists()
    metric_keys = sorted(agg.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["task", "adapter", "model", "seed", "n_eval"]
                            + metric_keys + ["checkpoint"])
        writer.writerow(
            [args.task, cfg["adapter"]["type"], cfg["model"]["name"],
             args.seed, len(results)]
            + [f"{agg[k]:.4f}" for k in metric_keys]
            + [str(args.checkpoint)]
        )

    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
