#!/bin/bash
# Two-stage ReGenHuman LoRA training (DiffSynth "split training").
#
#   Stage 1: run the VAE + T5 once per sample and cache the results to disk.
#   Stage 2: train only the DiT/VACE LoRA from that cache.
#
# The single-stage train_lora.sh redoes the VAE (2 videos) and T5 encode inside
# every training step. Here that cost is paid once per sample.
#
# Usage :
#   DATASET_DIR=data/dataset_for_regenhuman/structhuman \
#   OUTPUT_DIR=outputs/structhuman_lora_split \
#   EVERY=4 EPOCHS=2 SAVE_STEPS=500 \
#     bash scripts/split_training.sh
#
# Variables (defaults in brackets):
#   BASE_REPO [Wan-AI/Wan2.1-VACE-1.3B]   set Wan-AI/Wan2.1-VACE-14B for the 14B
#   EVERY [1]        use every Nth row of metadata_vace.csv (neighbouring segments are near-duplicates)
#   EPOCHS [1]       passes over the (thinned) dataset in stage 2
#   SAVE_STEPS [500] checkpoint every N steps
#   WORKERS [4]      dataloader workers (DiffSynth default is 0 = decode in the main process)
#   GPUS [1]  LR [1e-4]
#
# Re-running skips stage 1 if its cache is complete. Delete $OUTPUT_DIR/cache/.done to redo it.
# Requires the PATCHED DiffSynth clone (scripts/apply_diffsynth_patch.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-$REPO_ROOT/third_party/DiffSynth-Studio}"
DATASET_DIR="${DATASET_DIR:?set DATASET_DIR to the prepared dataset (contains metadata_vace.csv)}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
BASE_REPO="${BASE_REPO:-Wan-AI/Wan2.1-VACE-1.3B}"
EVERY="${EVERY:-1}"
EPOCHS="${EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
WORKERS="${WORKERS:-4}"
GPUS="${GPUS:-1}"
LR="${LR:-1e-4}"

DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"; OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
CACHE_DIR="$OUTPUT_DIR/cache"
LORA_DIR="$OUTPUT_DIR/lora"
mkdir -p "$CACHE_DIR" "$LORA_DIR"

[ -f "$DATASET_DIR/metadata_vace.csv" ] || { echo "ERROR: $DATASET_DIR/metadata_vace.csv not found"; exit 1; }
[ -d "$DIFFSYNTH_DIR" ] || { echo "ERROR: patched DiffSynth-Studio not found at $DIFFSYNTH_DIR"; exit 1; }

# optional thinning (prompts are single-line, so one CSV row == one line)
META="$DATASET_DIR/metadata_vace.csv"
if [ "$EVERY" -gt 1 ]; then
    META="$OUTPUT_DIR/metadata_every${EVERY}.csv"
    awk -v n="$EVERY" 'NR==1 || (NR-2) % n == 0' "$DATASET_DIR/metadata_vace.csv" > "$META"
fi
N_SAMPLES=$(( $(wc -l < "$META") - 1 ))
echo "samples: $N_SAMPLES | epochs: $EPOCHS | total stage-2 steps: $(( N_SAMPLES * EPOCHS / GPUS )) | base: $BASE_REPO"

mkdir -p "$REPO_ROOT/models"
[ -e "$DIFFSYNTH_DIR/models" ] || ln -s "$REPO_ROOT/models" "$DIFFSYNTH_DIR/models"

DIT="$BASE_REPO:diffusion_pytorch_model*.safetensors"   # wildcard: the 14B DiT is sharded
T5="$BASE_REPO:models_t5_umt5-xxl-enc-bf16.pth"
VAE="$BASE_REPO:Wan2.1_VAE.pth"
ALL_MODELS="$DIT,$T5,$VAE"

cd "$DIFFSYNTH_DIR"
export PYTHONPATH="$DIFFSYNTH_DIR:${PYTHONPATH:-}"

COMMON=(
    --height 480 --width 832 --num_frames 49
    --dataset_repeat 1
    --dataset_num_workers "$WORKERS"
    --model_id_with_origin_paths "$ALL_MODELS"
    --learning_rate "$LR"
    --num_epochs "$EPOCHS"
    --remove_prefix_in_ckpt "pipe.vace."
    --lora_base_model "vace"
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
    --lora_rank 32
    --extra_inputs "vace_video,vace_reference_image"
)

if [ ! -f "$CACHE_DIR/.done" ]; then
    echo "=== stage 1: caching VAE/T5 results to $CACHE_DIR ==="
    df -h "$CACHE_DIR" | tail -1
    accelerate launch --num_processes "$GPUS" --mixed_precision bf16 \
        examples/wanvideo/model_training/train.py \
        --dataset_base_path "$DATASET_DIR" \
        --dataset_metadata_path "$META" \
        --data_file_keys "video,vace_video,vace_reference_image" \
        --output_path "$CACHE_DIR" \
        --task "sft:data_process" \
        --offload_models "$DIT" \
        "${COMMON[@]}"
    touch "$CACHE_DIR/.done"
else
    echo "=== stage 1: cache already complete, skipping ==="
fi

echo "=== stage 2: training the LoRA ==="
accelerate launch --num_processes "$GPUS" --mixed_precision bf16 \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "$CACHE_DIR" \
    --save_steps "$SAVE_STEPS" \
    --output_path "$LORA_DIR" \
    --task "sft:train" \
    --offload_models "$T5,$VAE" \
    "${COMMON[@]}" \
    "$@"

echo "done. LoRA checkpoints: $LORA_DIR"