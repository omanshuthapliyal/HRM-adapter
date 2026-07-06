#!/usr/bin/env bash
# run_parity_sweep.sh -- Long-horizon parity sweep (Task 1)
#
# For each T in {32, 64, 128, 256, 512}:
#   1. Train LoRA model from scratch on long_parity (all params free)
#   2. Train HRM model from scratch on long_parity (all params free)
#   3. BT-reduce the HRM model
#   4. Zero-shot eval of BT-reduced model
#
# Expected result: LoRA saturates ~T=128-256; HRM continues scaling.
#
# Prerequisites:
#   conda activate hrm-mac
#   export PYTHONPATH=$(pwd)
#
# Usage:
#   bash scripts/run_parity_sweep.sh
#   bash scripts/run_parity_sweep.sh 32 64    # run subset

set -euo pipefail

EPOCHS=50
BATCH=64
N_TRAIN=5000
N_VAL=500
SEED=42
LORA_RANK=8
STATE_DIM=32
HSV_THRESHOLD=0.01

T_VALUES="${@:-32 64 128 256 512}"

# Results file -- one CSV row per (T, adapter)
RESULTS_CSV="logs/parity_sweep_results.csv"
echo "T,adapter,val_acc,d_hat_l0,d_hat_l1" > "$RESULTS_CSV"

echo "============================================================"
echo "  Long-Horizon Parity Sweep"
echo "  T values: $T_VALUES"
echo "  Epochs: $EPOCHS  Batch: $BATCH"
echo "============================================================"
echo ""

for T in $T_VALUES; do
    echo "------------------------------------------------------------"
    echo "  T = $T  (model seq_len = $((2*T+1)))"
    echo "------------------------------------------------------------"

    # --- LoRA from scratch ---
    echo "[sweep] LoRA from scratch, T=$T ..."
    python scripts/train_sweep.py \
        --task long_parity \
        --T "$T" \
        --adapter lora \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --n_train "$N_TRAIN" \
        --n_val "$N_VAL" \
        --lora_rank "$LORA_RANK" \
        --seed "$SEED" 2>&1 | tee /tmp/lora_out_T${T}.txt | tail -5

    LORA_ACC=$(grep -oE 'val_acc=[0-9.]+' /tmp/lora_out_T${T}.txt | tail -1 | cut -d= -f2)
    echo "$T,lora,$LORA_ACC,NA,NA" >> "$RESULTS_CSV"
    echo "[parity] LoRA T=$T  acc=$LORA_ACC"

    # --- HRM from scratch ---
    echo "[sweep] HRM from scratch, T=$T ..."
    python scripts/train_sweep.py \
        --task long_parity \
        --T "$T" \
        --adapter hrm \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --n_train "$N_TRAIN" \
        --n_val "$N_VAL" \
        --state_dim "$STATE_DIM" \
        --seed "$SEED" 2>&1 | tee /tmp/hrm_out_T${T}.txt | tail -5

    HRM_ACC=$(grep -oE 'val_acc=[0-9.]+' /tmp/hrm_out_T${T}.txt | tail -1 | cut -d= -f2)
    echo "$T,hrm,$HRM_ACC,NA,NA" >> "$RESULTS_CSV"
    echo "[parity] HRM  T=$T  acc=$HRM_ACC"

    # --- BT Reduce ---
    HRM_CKPT="checkpoints/sweep_long_parity_T${T}_hrm/best.pt"
    echo "[sweep] BT reduce T=$T ..."
    python scripts/reduce_adapter.py \
        --load "$HRM_CKPT" \
        --task "long_parity_T${T}" \
        --state_dim "$STATE_DIM" \
        --hsv_threshold "$HSV_THRESHOLD" 2>&1 | tee /tmp/reduce_out_T${T}.txt | grep -E "Truncation|Reduction summary|Saved"

    D0=$(grep -oE 'layers\.0.*-> [0-9]+' /tmp/reduce_out_T${T}.txt | grep -oE '-> [0-9]+' | head -1 | grep -oE '[0-9]+')
    D1=$(grep -oE 'layers\.1.*-> [0-9]+' /tmp/reduce_out_T${T}.txt | grep -oE '-> [0-9]+' | head -1 | grep -oE '[0-9]+')
    D0="${D0:-?}"; D1="${D1:-?}"

    # --- HRM Phase 2 zero-shot ---
    REDUCED_CKPT="checkpoints/hrm_reduced_long_parity_T${T}/model_reduced.pt"
    echo "[sweep] HRM-BT zero-shot T=$T ..."
    python scripts/train_hrm_phase2.py \
        --load "$REDUCED_CKPT" \
        --task long_parity \
        --task_T "$T" \
        --batch "$BATCH" \
        --n_train "$N_TRAIN" \
        --n_val "$N_VAL" \
        --seed "$SEED" 2>&1 | tee /tmp/p2_out_T${T}.txt | grep -E "Zero-shot|Step 5"

    BT_ACC=$(grep -oE 'Zero-shot.*acc=[0-9.]+' /tmp/p2_out_T${T}.txt | grep -oE 'acc=[0-9.]+' | cut -d= -f2)
    BT_ACC="${BT_ACC:-NA}"
    echo "$T,hrm_bt,$BT_ACC,$D0,$D1" >> "$RESULTS_CSV"
    echo "[parity] HRM-BT T=$T  acc=$BT_ACC  d_hat=$D0/$D1"
    echo ""
done

# -----------------------------------------------------------------------
echo "============================================================"
echo "  PARITY SWEEP RESULTS"
echo "============================================================"
python3 - <<'EOF'
import csv, sys
rows = list(csv.DictReader(open("logs/parity_sweep_results.csv")))
Ts = sorted(set(r["T"] for r in rows), key=int)
lora = {r["T"]: r["val_acc"] for r in rows if r["adapter"] == "lora"}
hrm  = {r["T"]: r["val_acc"] for r in rows if r["adapter"] == "hrm"}
bt   = {r["T"]: r["val_acc"] for r in rows if r["adapter"] == "hrm_bt"}
dhat = {r["T"]: f"{r['d_hat_l0']}/{r['d_hat_l1']}" for r in rows if r["adapter"] == "hrm_bt"}
print(f"{'T':>6}  {'LoRA(r=8)':>10}  {'HRM(d=32)':>10}  {'HRM-BT':>10}  {'d_hat(L0/L1)':>14}  {'HRM>LoRA':>9}")
print("-" * 70)
wins = 0
for T in Ts:
    l = lora.get(T,"N/A"); h = hrm.get(T,"N/A")
    b = bt.get(T,"N/A");   d = dhat.get(T,"?/?")
    try:
        win = "YES" if float(h) > float(l) else "no"
        if win == "YES": wins += 1
    except: win = "?"
    print(f"{T:>6}  {l:>10}  {h:>10}  {b:>10}  {d:>14}  {win:>9}")
print(f"\nSSM advantage: HRM > LoRA on {wins}/{len(Ts)} lengths")
EOF
