# SSM Adapters via Hankel Reduced-order Modeling

Code for the paper:

> **SSM Adapters via Hankel Reduced-order Modeling: Injection Site Determines Task Suitability in Long-Context Fine-Tuning**
> Omanshu Thapliyal
> 4th Workshop on High-dimensional Learning Dynamics (HiLD), 43rd International Conference on Machine Learning, 2026.
> arXiv: https://arxiv.org/abs/2606.26290

## Overview

Parameter-efficient fine-tuning (PEFT) methods such as LoRA target attention projectors. Their
adapter output at each position is a static function of the current token only -- they cannot
maintain state across positions. This limits their effectiveness on tasks that require sequential
state accumulation (e.g., reading comprehension over long documents, meeting summarization, formal
language tracking).

This paper introduces the **Hankel Reduced-order Model (HRM) adapter**: an SSM-based residual
module injected parallel to the MLP sub-layer of a frozen transformer. The key contributions are:

1. **Injection site as inductive bias.** Attention injection enables positional retrieval; MLP
   injection enables sequential integration. The injection site is the primary determinant of
   task suitability, not the adapter design within the site.

2. **Provable compression via Balanced Truncation.** HRM is initialized by computing empirical
   Hankel Grammians of a trained SSM adapter and performing balanced truncation. This yields a
   reduced system with a certified H-infinity error bound (Glover, 1984). The Hankel singular
   value (HSV) spectrum is a data-driven fingerprint of the task's temporal complexity.

3. **Computational parity with LoRA.** Because the state transition matrix A is time-invariant
   and diagonal, the HRM recurrence is a causal linear convolution computable via FFT in
   O(T log T), matching LoRA's compute at all context lengths.

In iso-parametric evaluations on Mistral-7B (8.4M trainable parameters), HRM outperforms all
LoRA variants on tasks requiring sequential evidence integration: QuALITY (+34.8% relative
accuracy) and QMSum (+71.6% relative ROUGE-1). HRM also achieves lower BPC than LoRA across all
18 configurations on enwiki8 and all T values on MAESTRO piano language modeling.

## Repository Layout

```
hrm-adapters/
  src/                      Core library (backbone + adapter + BT pipeline)
    models/
      transformer.py        TinyGPT backbone (2-4 layer, d_model configurable)
      ssm.py                Diagonal SSM with ZOH discretization and FFT scan
    adapters/
      hrm_adapter.py        HRM adapter: gated SSM residual parallel to MLP
      lora.py               LoRA baseline adapter
      insertion.py          inject_lora() / inject_hrm() in-place into TinyGPT
    reduction/
      grammians.py          Exact and empirical Hankel Grammians
      balanced_truncation.py  BT reduction with Glover H-infinity error bound
      hooks.py              Forward hooks for hidden state snapshot collection
      stability.py          Eigenvalue stability check and projection
    data/
      recurrent_tasks.py    DFA state tracking, Parity, and related tasks
      enwiki8.py            enwiki8 character-level language modeling
      maestro.py            MAESTRO piano MIDI language modeling
    training/
      trainer.py            Training loop with masked cross-entropy
      metrics.py            Accuracy and BPC metrics
  scripts/                  Entry points for synthetic task experiments
    train_sweep.py          Train HRM or LoRA from scratch on any task
    train_hrm_phase1.py     Phase 1: train full-rank SSM adapter
    reduce_adapter.py       Compute Grammians, run BT, save reduced model
    train_hrm_phase2.py     Phase 2: fine-tune with BT-reduced adapter
    train_lora_baseline.py  LoRA baseline training
    run_dfa_sweep.sh        DFA state tracking sweep (Section 5.1)
    run_parity_sweep.sh     Long-horizon parity sweep
    run_enwiki8_sweep.sh    enwiki8 character LM sweep (Section 6.1)
    run_maestro_sweep.sh    MAESTRO piano LM sweep (Section 6)
  configs/                  YAML configs for synthetic task experiments
  mistral7b/                Mistral-7B LongBench experiments (Section 7)
    src/                    HRM and injection code for HuggingFace models
    scripts/
      train_peft_longbench.py  PEFT fine-tuning on QuALITY, QMSum, NarrativeQA
      eval_longbench.py        Evaluation on LongBench tasks
      reduce_mistral_hrm.py    BT reduction for Mistral-7B HRM checkpoints
    configs/                   Per-method YAML configs for Mistral-7B
```

## Installation

**Mac (Apple Silicon) -- local prototyping:**
```bash
conda env create -f environment_mac.yaml
conda activate hrm-mac
export PYTHONPATH=$(pwd)
```
PyTorch uses the MPS backend. If an op is unsupported, prefix with:
`PYTORCH_ENABLE_MPS_FALLBACK=1`

**GPU server (CUDA) -- full experiments:**
```bash
conda env create -f environment_cuda.yaml
conda activate hrm-gpu
export PYTHONPATH=$(pwd)
```

**Sanity check:**
```bash
make smoke
```

**Mistral-7B additional dependencies:**
```bash
pip install -r mistral7b/requirements.txt
```

## Data

**enwiki8** (character-level Wikipedia, 100M bytes):
```bash
mkdir -p data/enwiki8
curl -L http://mattmahoney.net/dc/enwik8.zip -o data/enwiki8/enwik8.zip
unzip data/enwiki8/enwik8.zip -d data/enwiki8/
```

**MAESTRO v2** (piano MIDI, ~200 hours):
Download from https://magenta.tensorflow.org/datasets/maestro and place at `data/maestro-v3.0.0/`.

**DFA / Parity:** Generated on the fly -- no download required.

**LongBench (QuALITY, QMSum, NarrativeQA):** Loaded automatically from HuggingFace Hub
(`emozilla/quality`, `pszemraj/qmsum-cleaned`, `deepmind/narrativeqa`) by
`mistral7b/scripts/train_peft_longbench.py`.

## Reproducing Experiments

### DFA State Tracking (Section 5.1, Figure 2)

Trains HRM and LoRA from scratch on DFA state prediction with k in {2, 4, 8} states and
context lengths T in {64, 128, 256, 512}. BT reduction is applied at eps=0.01.

```bash
conda activate hrm-mac
export PYTHONPATH=$(pwd)
bash scripts/run_dfa_sweep.sh
# Results written to logs/dfa_sweep_results.csv
```

### MAESTRO Piano Language Modeling (Section 6, Figure 3)

```bash
bash scripts/run_maestro_sweep.sh
# Requires CUDA; backbone: d_model=256, 4 layers
# Results written to logs/maestro_sweep_results.csv
```

### enwiki8 Character Language Modeling (Section 6.1, Table 3)

Three iso-parametric tiers (LoRA rank r in {8, 16, 32} matched to HRM state dim d in {16, 32, 63})
across context lengths T in {512, 1024, 2048}:

```bash
bash scripts/run_enwiki8_sweep.sh               # all 3 tiers, seed=42
bash scripts/run_enwiki8_sweep.sh --multi_seed  # 3 seeds (longer)
# Results written to logs/enwiki8_sweep_results.csv
```

### Balanced Truncation Pipeline (Section 4, Equations 11-14)

Phase 1 trains a full-rank SSM adapter. `reduce_adapter.py` computes exact analytic Grammians
(diagonal-A closed form), performs balanced truncation, and saves the reduced system matrices
with a Glover H-infinity error bound. Phase 2 fine-tunes the reduced adapter.

```bash
# Phase 1: train SSM adapter
python scripts/train_hrm_phase1.py --config configs/adapter/hrm.yaml \
  --task dfa --T 128 --seed 42

# Reduce: compute Grammians + BT
python scripts/reduce_adapter.py \
  --load checkpoints/hrm_phase1/best.pt \
  --task dfa --hsv_threshold 0.01

# Phase 2: fine-tune reduced adapter
python scripts/train_hrm_phase2.py --config configs/adapter/hrm.yaml \
  --load checkpoints/hrm_reduced_dfa_T128/model_reduced.pt \
  --task dfa --T 128 --seed 42
```

### Mistral-7B LongBench (Section 7, Table 5)

Requires 2x NVIDIA RTX 4090 (48 GB total VRAM) or equivalent. All methods use 8.4M trainable
parameters (HRM state dim d=32, LoRA rank r=16).

```bash
cd mistral7b

# HRM (parallel to MLP)
python scripts/train_peft_longbench.py \
  --config configs/hrm_mistral7b.yaml \
  --task quality --seed 42 \
  --output_dir logs/mistral7b_hrm_quality_s42

# LoRA (attention projectors)
python scripts/train_peft_longbench.py \
  --config configs/lora_mistral7b.yaml \
  --task quality --seed 42 \
  --output_dir logs/mistral7b_lora_quality_s42

# Evaluate
python scripts/eval_longbench.py \
  --checkpoint logs/mistral7b_hrm_quality_s42 \
  --config configs/hrm_mistral7b.yaml \
  --task quality \
  --output_dir logs/eval_hrm_quality_s42
```

Other adapters: substitute `lora_mistral7b.yaml`, `adalora_mistral7b.yaml`,
`dora_mistral7b.yaml`, `qlora_mistral7b.yaml`, or `mlp_lora_mistral7b.yaml`.

**BT reduction for Mistral-7B:**
```bash
python scripts/reduce_mistral_hrm.py \
  --task quality --seed 42 --eps 0.10 \
  --log_dir logs/

python scripts/train_peft_longbench.py \
  --config configs/hrm_mistral7b.yaml \
  --task quality --seed 42 \
  --bt_reduce_init logs/bt_reduced_quality_s42_eps10.pt \
  --output_dir logs/mistral7b_hrm_bt_quality_s42
```

## Key Results

**DFA state tracking (4 states, seed=42):**

| T   | LoRA  | HRM   | HRM-BT |
|-----|-------|-------|--------|
| 64  | 0.643 | 0.832 | 0.831  |
| 128 | 0.476 | 0.871 | 0.868  |
| 256 | 0.342 | 0.856 | 0.854  |
| 512 | 0.291 | 0.843 | 0.840  |

**Mistral-7B LongBench (8.4M trainable params, seed=42):**

| Method       | QuALITY (acc) | QMSum (R-1) |
|--------------|---------------|-------------|
| LoRA         | 0.3518        | 0.1477      |
| AdaLoRA      | 0.3518        | 0.1477      |
| DoRA         | 0.3478        | 0.1458      |
| QLoRA        | 0.3399        | 0.1434      |
| HRM (ours)   | **0.4743**    | **0.2531**  |

**enwiki8 (all 18 iso-parametric configs, Tier 2, T=1024, seed=42):**
HRM achieves lower BPC than LoRA in all 18 of 18 configurations tested (p ~= 3.8e-6, sign test).

## HRM Adapter Architecture

The HRM adapter is injected parallel to the MLP sub-layer of each transformer block. For layer l
with MLP output h_MLP at position t:

```
h_out = h_MLP + alpha * y_t
s_t   = A_bar * s_{t-1} + B_bar @ h_MLP    (SSM recurrence)
y_t   = C @ s_t
```

where A_bar = diag(exp(-exp(log_A) * delta)), B_bar and C are dense matrices, and alpha is a
per-layer learnable scalar gate initialized to 0 (zero-residual init). The time-invariant
diagonal A enables exact FFT-based evaluation in O(T log T).

After training, Balanced Truncation reduces the state dimension d to d_hat < d while bounding
the approximation error: ||G - G_hat||_{H_inf} <= 2 * sum_{i > d_hat} sigma_i, where sigma_i
are the Hankel singular values.

## Citation

```bibtex
@inproceedings{thapliyal2026hrm,
  title     = {{SSM} Adapters via {Hankel} Reduced-order Modeling:
               Injection Site Determines Task Suitability in
               Long-Context Fine-Tuning},
  author    = {Thapliyal, Omanshu},
  booktitle = {4th Workshop on High-dimensional Learning Dynamics ({HiLD}),
               43rd International Conference on Machine Learning},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.26290}
}
```

## License

MIT
