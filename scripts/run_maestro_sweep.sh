#!/usr/bin/env bash
# run_maestro_gpu_sweep.sh -- MAESTRO GPU sweep with larger backbone (2xRTX 4090)
#
# Backbone: TinyGPT d_model=256, n_heads=8, n_layers=4, d_ff=1024
# ISO-PARAMETER pairs recalculated for d_model=256:
#   HRM adapter params per layer: 516xd + 2  (input_dim=output_dim=256)
#   LoRA adapter params per layer: 4 x (rankx256 + 256xrank) = 2048xrank
#     (4 modules: q_proj+v_proj x 2 layers)
#
#   Tier 1: LoRA r=8  (16,384 params) ~= HRM d=31 (16,498 params) -- <1% diff
#   Tier 2: LoRA r=16 (32,768 params) ~= HRM d=63 (32,510 params) -- <1% diff
#   Tier 3: LoRA r=32 (65,536 params) ~= HRM d=127 (65,534 params) -- <0.01% diff
#
# Expected accuracy: 50-60% (vs 37-40% with d=128 backbone)
# Hardware: CUDA device auto-detected; uses DataLoader num_workers=4
#
# Usage:
#   conda activate hrm-gpu
#   export PYTHONPATH=$(pwd)
#   bash scripts/run_maestro_gpu_sweep.sh
#   bash scripts/run_maestro_gpu_sweep.sh --maestro_dir /path/to/maestro-v3.0.0

set -euo pipefail

MAESTRO_DIR="${MAESTRO_DIR:-data/maestro-v3.0.0}"
CSV="logs/maestro_gpu_sweep_results.csv"
EPOCHS=60
N_TRAIN=50000
N_VAL=5000
BATCH=128
LR=3e-4
SEEDS=(42 123 456)
TOTAL_RUNS=18   # 3 tiers x 2 adapters x 3 seeds

while [[ $# -gt 0 ]]; do
    case "$1" in
        --maestro_dir) MAESTRO_DIR="$2"; shift 2 ;;
        --seeds)       IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --single_seed) SEEDS=(42); TOTAL_RUNS=6; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$MAESTRO_DIR/maestro-v3.0.0.json" ]; then
    echo "ERROR: MAESTRO metadata not found at $MAESTRO_DIR/maestro-v3.0.0.json"
    exit 1
fi

# Verify CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found -- wrong env?'; print(f'CUDA: {torch.cuda.get_device_name(0)}')"

mkdir -p logs

if [ ! -f "$CSV" ]; then
    echo "task,T,adapter,config_key,config_val,val_acc,trainable_params,seed,epoch_time_s" > "$CSV"
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
    local suffix="maestro_gpu_${adapter}_$(echo "$config_key" | tr _ -)${config_val}_T${T}_s${seed}"

    if grep -q "^maestro_gpu,${T},${adapter},${config_key},${config_val}," "$CSV" 2>/dev/null; then
        if grep "^maestro_gpu,${T},${adapter},${config_key},${config_val}," "$CSV" | grep -q ",${seed},\|,${seed}$"; then
            echo ">>> SKIP (done)  T=${T}  adapter=${adapter}  ${config_key}=${config_val}  seed=${seed}"
            DONE=$(( DONE + 1 ))
            return
        fi
    fi

    DONE=$(( DONE + 1 ))
    NOW=$(date +%s); ELAPSED=$(( NOW - START_TS ))
    if [ "$DONE" -gt 1 ]; then
        ETA=$(( ELAPSED * (TOTAL_RUNS - DONE + 1) / (DONE - 1) ))
    else ETA=0; fi
    echo ">>> [${DONE}/${TOTAL_RUNS}] GPU  T=${T}  adapter=${adapter}  ${config_key}=${config_val}  seed=${seed}  ETA=$(fmt_dur $ETA)"

    tmpfile=$(mktemp /tmp/maestro_gpu_XXXXXX)
    python scripts/train_sweep.py \
        --task maestro --T "$T" --adapter "$adapter" \
        --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
        --n_train "$N_TRAIN" --n_val "$N_VAL" \
        --seed "$seed" --run_suffix "$suffix" \
        --maestro_dir "$MAESTRO_DIR" \
        --d_model 256 --n_layers 4 --n_heads 8 --d_ff 1024 \
        $extra_args 2>&1 | tee "$tmpfile"

    val_acc=$(grep "RESULT" "$tmpfile" | grep -oE 'val_acc=[0-9.]+' | tail -1 | sed 's/val_acc=//' || true)
    trainable=$(grep -oE "'trainable': [0-9]+" "$tmpfile" | tail -1 | grep -oE '[0-9]+$' || true)
    epoch_time=$(grep -oE 'epoch_time_s=[0-9.]+' "$tmpfile" | tail -1 | sed 's/epoch_time_s=//' || true)
    val_acc="${val_acc:-NA}"
    trainable="${trainable:-NA}"
    epoch_time="${epoch_time:-NA}"

    echo "maestro_gpu,${T},${adapter},${config_key},${config_val},${val_acc},${trainable},${seed},${epoch_time}" >> "$CSV"
    rm -f "$tmpfile"
}

echo "=============================================================="
echo "  MAESTRO GPU Sweep -- d=256, n_layers=4"
echo "  ${TOTAL_RUNS} runs | seeds=${SEEDS[*]} | dir=${MAESTRO_DIR}"
echo "  Started: $(date)"
echo "=============================================================="

for seed in "${SEEDS[@]}"; do
    # Tier 1: LoRA r=8 (16,384) ~= HRM d=31 (16,498) -- <1%
    run_one 512 "lora" "lora_rank" "8"  "$seed" "--lora_rank 8"
    run_one 512 "hrm"  "state_dim" "31" "$seed" "--state_dim 31 --gate_init 0.1"
    # Tier 2: LoRA r=16 (32,768) ~= HRM d=63 (32,510) -- <1%
    run_one 512 "lora" "lora_rank" "16" "$seed" "--lora_rank 16"
    run_one 512 "hrm"  "state_dim" "63" "$seed" "--state_dim 63 --gate_init 0.1"
    # Tier 3: LoRA r=32 (65,536) ~= HRM d=127 (65,534) -- near exact match
    run_one 512 "lora" "lora_rank" "32" "$seed" "--lora_rank 32"
    run_one 512 "hrm"  "state_dim" "127" "$seed" "--state_dim 127 --gate_init 0.1"
done

echo ""
echo "=============================================================="
echo "  GPU sweep complete: $(date)"
echo "  Rows: $(( $(wc -l < "$CSV") - 1 ))"
echo "  Results: ${CSV}"
echo "=============================================================="
