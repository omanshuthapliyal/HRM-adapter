#!/usr/bin/env bash
# run_enwiki8_sweep.sh -- enwiki8 character-level LM sweep (long-range HRM vs LoRA)
#
# Tests whether SSM recurrence (HRM) overtakes static LoRA at longer context windows.
# Expected: HRM advantage grows with T -- large gap at T=2048 where LoRA's static deltaW
# cannot aggregate state across the full context window.
#
# Backbone: TinyGPT d_model=128, n_heads=4, n_layers=4, d_ff=512  (4-layer for LM)
#
# ISO-PARAMETER pairs (n_layers=4, d_model=128):
#   HRM params per adapter: 258d+1 (B:dx128, C:128xd, log_A:d, log_dt:d, gate:1)
#   With 4 layers: 4x(258d+1) = 1032d+4
#   LoRA params (q+v, 4 layers): 4x2x2xrx128 = 2048r
#
#   Tier 1: LoRA r=8  (16,384) ~= HRM d=16 (16,516) -- diff=0.8% 
#   Tier 2: LoRA r=16 (32,768) ~= HRM d=32 (33,028) -- diff=0.8% 
#   Tier 3: LoRA r=32 (65,536) ~= HRM d=63 (65,020) -- diff=0.8% 
#
# Metric: BPC (bits-per-character) = cross_entropy_nats / ln(2); lower is better.
# Also reports top-1 NTP accuracy (val_acc).
#
# Data download (before running):
#   mkdir -p data/enwiki8
#   curl -L http://mattmahoney.net/dc/enwik8.zip -o data/enwiki8/enwik8.zip
#   unzip data/enwiki8/enwik8.zip -d data/enwiki8/
#
# Usage:
#   conda activate hrm-mac
#   export PYTHONPATH=$(pwd)
#   bash scripts/run_enwiki8_sweep.sh                     # Tier 2, Tin{512,1024,2048}, seed=42
#   bash scripts/run_enwiki8_sweep.sh --all_tiers         # all 3 tiers
#   bash scripts/run_enwiki8_sweep.sh --multi_seed        # 3 seeds (overnight)
#   bash scripts/run_enwiki8_sweep.sh --data_dir /path    # custom data dir

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/enwiki8}"
CSV="logs/enwiki8_sweep_results.csv"
EPOCHS=25
N_TRAIN=10000
N_VAL=1000
BATCH=32
LR=3e-4
SEEDS=(42)
TIERS="1,2,3"       # default: all tiers
T_VALUES=(512 1024 2048)
TOTAL_RUNS=18       # default: 3 tiers x 2 adapters x 3 T x 1 seed

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)   DATA_DIR="$2"; shift 2 ;;
        --multi_seed) SEEDS=(42 123 456); shift ;;
        --tiers)      TIERS="$2"; shift 2 ;;
        --t_values)   IFS=',' read -ra T_VALUES <<< "$2"; shift 2 ;;
        --n_train)    N_TRAIN="$2"; shift 2 ;;
        --n_val)      N_VAL="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --lr)         LR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Recalculate TOTAL_RUNS
n_seeds=${#SEEDS[@]}
n_tiers=$(echo "$TIERS" | tr ',' '\n' | wc -l | tr -d ' ')
n_t=${#T_VALUES[@]}
TOTAL_RUNS=$(( n_tiers * 2 * n_t * n_seeds ))

if [ ! -f "$DATA_DIR/enwik8" ]; then
    echo "ERROR: enwik8 file not found at $DATA_DIR/enwik8"
    echo "Download with:"
    echo "  mkdir -p $DATA_DIR"
    echo "  curl -L http://mattmahoney.net/dc/enwik8.zip -o $DATA_DIR/enwik8.zip"
    echo "  unzip $DATA_DIR/enwik8.zip -d $DATA_DIR/"
    exit 1
fi

mkdir -p logs

if [ ! -f "$CSV" ]; then
    echo "task,T,adapter,config_key,config_val,val_acc,val_bpc,trainable_params,seed,epoch_time_s" > "$CSV"
fi

DONE=0
START_TS=$(date +%s)

fmt_dur() {
    local s=$1
    if   [ "$s" -lt 60 ];   then printf "%ds" "$s"
    elif [ "$s" -lt 3600 ]; then printf "%dm%02ds" $((s/60)) $((s%60))
    else                         printf "%dh%02dm" $((s/3600)) $(( (s%3600)/60 ))
    fi
}

run_one() {
    local T="$1" adapter="$2" config_key="$3" config_val="$4" seed="$5" extra_args="$6"
    local suffix="enwiki8_${adapter}_$(echo "$config_key" | tr _ -)${config_val}_T${T}_s${seed}"

    # Skip if already done
    if grep -qE "^enwiki8,${T},${adapter},${config_key},${config_val},[^,]*,[^,]*,[^,]*,${seed}," "$CSV" 2>/dev/null; then
        echo ">>> SKIP (done)  T=${T}  adapter=${adapter}  ${config_key}=${config_val}  seed=${seed}"
        DONE=$(( DONE + 1 ))
        return
    fi

    DONE=$(( DONE + 1 ))
    NOW=$(date +%s); ELAPSED=$(( NOW - START_TS ))
    if [ "$DONE" -gt 1 ]; then
        ETA=$(( ELAPSED * (TOTAL_RUNS - DONE + 1) / (DONE - 1) ))
    else ETA=0; fi
    echo ">>> [${DONE}/${TOTAL_RUNS}] enwiki8  T=${T}  adapter=${adapter}  ${config_key}=${config_val}  seed=${seed}  ETA=$(fmt_dur $ETA)"

    tmpfile=$(mktemp /tmp/enwiki8_XXXXXX)
    python scripts/train_sweep.py \
        --task enwiki8 --T "$T" --adapter "$adapter" \
        --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
        --n_train "$N_TRAIN" --n_val "$N_VAL" \
        --seed "$seed" --run_suffix "$suffix" \
        --data_dir "$DATA_DIR" \
        --n_layers 4 \
        $extra_args 2>&1 | tee "$tmpfile"

    val_acc=$(grep "RESULT" "$tmpfile" | grep -oE 'val_acc=[0-9.]+' | tail -1 | sed 's/val_acc=//' || true)
    val_bpc=$(grep "RESULT" "$tmpfile" | grep -oE 'val_bpc=[0-9.]+' | tail -1 | sed 's/val_bpc=//' || true)
    trainable=$(grep -oE "'trainable': [0-9]+" "$tmpfile" | tail -1 | grep -oE '[0-9]+$' || true)
    epoch_time=$(grep -oE 'epoch_time_s=[0-9.]+' "$tmpfile" | tail -1 | sed 's/epoch_time_s=//' || true)
    val_acc="${val_acc:-NA}"
    val_bpc="${val_bpc:-NA}"
    trainable="${trainable:-NA}"
    epoch_time="${epoch_time:-NA}"

    echo "enwiki8,${T},${adapter},${config_key},${config_val},${val_acc},${val_bpc},${trainable},${seed},${epoch_time}" >> "$CSV"
    rm -f "$tmpfile"
}

echo "=============================================================="
echo "  enwiki8 LM Sweep -- 4-layer TinyGPT, d_model=128"
echo "  Tiers: ${TIERS} | T: ${T_VALUES[*]} | seeds: ${SEEDS[*]}"
echo "  ${TOTAL_RUNS} total runs | data: ${DATA_DIR}"
echo "  Started: $(date)"
echo "=============================================================="

for seed in "${SEEDS[@]}"; do
    for T in "${T_VALUES[@]}"; do
        if echo "$TIERS" | grep -q "1"; then
            # Tier 1: LoRA r=8 (16,384) ~= HRM d=16 (16,516) -- 0.8% diff
            run_one "$T" "lora" "lora_rank" "8"  "$seed" "--lora_rank 8"
            run_one "$T" "hrm"  "state_dim" "16" "$seed" "--state_dim 16 --gate_init 0.1"
        fi
        if echo "$TIERS" | grep -q "2"; then
            # Tier 2: LoRA r=16 (32,768) ~= HRM d=32 (33,028) -- 0.8% diff
            run_one "$T" "lora" "lora_rank" "16" "$seed" "--lora_rank 16"
            run_one "$T" "hrm"  "state_dim" "32" "$seed" "--state_dim 32 --gate_init 0.1"
        fi
        if echo "$TIERS" | grep -q "3"; then
            # Tier 3: LoRA r=32 (65,536) ~= HRM d=63 (65,020) -- 0.8% diff
            run_one "$T" "lora" "lora_rank" "32" "$seed" "--lora_rank 32"
            run_one "$T" "hrm"  "state_dim" "63" "$seed" "--state_dim 63 --gate_init 0.1"
        fi
    done
done

echo ""
echo "=============================================================="
echo "  enwiki8 sweep complete: $(date)"
echo "  Rows: $(( $(wc -l < "$CSV") - 1 ))"
echo "  Results: ${CSV}"
echo "=============================================================="
