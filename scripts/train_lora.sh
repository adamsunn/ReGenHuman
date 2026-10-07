#!/bin/bash
# Train a ReGenHuman LoRA (rank 32 on the VACE context adapter) on a dataset
# built by scripts/prepare_dataset.py. This is the exact recipe of the
# released weights (lr 1e-4, 20 epochs, 480x832, 49 frames, bs 1/GPU).
#
# Usage:
#   DATASET_DIR=path/to/dataset/structall OUTPUT_DIR=outputs/structall_lora \
#     [GPUS=2] [DIFFSYNTH_DIR=third_party/DiffSynth-Studio] \
#     bash scripts/train_lora.sh
#
# Requires the PATCHED DiffSynth clone (scripts/apply_diffsynth_patch.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-$REPO_ROOT/third_party/DiffSynth-Studio}"
DATASET_DIR="${DATASET_DIR:?set DATASET_DIR to the prepared dataset (contains metadata_vace.csv)}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for checkpoints}"
GPUS="${GPUS:-1}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-20}"
SAVE_STEPS="${SAVE_STEPS:-2000}"

DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"; OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

if [ ! -f "$DATASET_DIR/metadata_vace.csv" ]; then
    echo "ERROR: $DATASET_DIR/metadata_vace.csv not found (run scripts/prepare_dataset.py)"; exit 1
fi
if [ ! -d "$DIFFSYNTH_DIR" ]; then
    echo "ERROR: patched DiffSynth-Studio not found at $DIFFSYNTH_DIR"
    echo "Run: bash scripts/apply_diffsynth_patch.sh"; exit 1
fi

# Base-model downloads resolve into ./models relative to the DiffSynth cwd;
# keep them inside the repo's models/ via a symlink.
mkdir -p "$REPO_ROOT/models"
[ -e "$DIFFSYNTH_DIR/models" ] || ln -s "$REPO_ROOT/models" "$DIFFSYNTH_DIR/models"

cd "$DIFFSYNTH_DIR"
export PYTHONPATH="$DIFFSYNTH_DIR:${PYTHONPATH:-}"

accelerate launch --num_processes "$GPUS" --mixed_precision bf16 \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "$DATASET_DIR" \
    --dataset_metadata_path "$DATASET_DIR/metadata_vace.csv" \
    --data_file_keys "video,vace_video,vace_reference_image" \
    --height 480 --width 832 --num_frames 49 \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
    --learning_rate "$LR" \
    --num_epochs "$EPOCHS" \
    --save_steps "$SAVE_STEPS" \
    --remove_prefix_in_ckpt "pipe.vace." \
    --output_path "$OUTPUT_DIR" \
    --lora_base_model "vace" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --extra_inputs "vace_video,vace_reference_image" \
    --use_gradient_checkpointing_offload \
    "$@"
