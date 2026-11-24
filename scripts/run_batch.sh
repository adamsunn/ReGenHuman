#!/usr/bin/env bash

set -Eeuo pipefail

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR="${1:?Usage: $0 INPUT_DIR COUNT [OUTPUT_DIR]}"
COUNT="${2:?Usage: $0 INPUT_DIR COUNT [OUTPUT_DIR]}"

OUTPUT_DIR="${3:-runs/batch_01}"
VARIANT="${VARIANT:-structhuman}"
SEED="${SEED:-}"

# Rate at which ReGenHuman generates (source frames are sub-sampled to this).
# 8 fps = stride 3 for 24 fps sources. Output is interpolated back to source fps.
GEN_FPS="${GEN_FPS:-24}"
# Optional override for diffusion steps (empty = demo.py default)
STEPS="${STEPS:-}"
# Set INTERPOLATE=0 to skip the ffmpeg interpolation step
INTERPOLATE="${INTERPOLATE:-0}"

PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LORA="${LORA:-$PROJECT_DIR/scripts/outputs/structhuman_lora_14b_resume/lora/step-4249.safetensors}"

cd "$PROJECT_DIR"

# --------------------------------------------------
# Directories
# --------------------------------------------------

mkdir -p \
    "$OUTPUT_DIR/prompts" \
    "$OUTPUT_DIR/logs" \
    "$OUTPUT_DIR/raw" \
    "$OUTPUT_DIR/final"

MANIFEST="$OUTPUT_DIR/manifest.txt"

# --------------------------------------------------
# Helpers
# --------------------------------------------------

# Prints "<total_frames> <fps>" for a video (fps as decimal)
probe_video() {
    local video="$1" total rate fps

    total="$(ffprobe -v error -select_streams v:0 -count_packets \
        -show_entries stream=nb_read_packets -of csv=p=0 "$video")"
    rate="$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate -of csv=p=0 "$video")"

    fps="$(awk -F/ -v r="$rate" 'BEGIN{split(r,a,"/"); print (a[2]>0 ? a[1]/a[2] : a[1])}')"

    if [[ -z "$total" || "$total" -le 0 || -z "$fps" ]]; then
        return 1
    fi
    echo "$total $fps"
}

# --------------------------------------------------
# 1. Sample videos
# --------------------------------------------------

echo "Sampling $COUNT videos from: $INPUT_DIR"

SAMPLER_ARGS=(
    --input-dir "$INPUT_DIR"
    --count "$COUNT"
)

if [[ -n "$SEED" ]]; then
    SAMPLER_ARGS+=(--seed "$SEED")
fi

"$PYTHON" scripts/sample_videos.py \
    "${SAMPLER_ARGS[@]}" > "$MANIFEST"

echo "Selected videos:"
cat "$MANIFEST"

# --------------------------------------------------
# 2. Generate prompts for ALL videos (model loaded once).
#    The process exits afterwards, freeing the GPU for ReGenHuman.
# --------------------------------------------------

echo
echo "Generating prompts with Qwen3-VL ..."

PROMPT_LOG="$OUTPUT_DIR/logs/prompt_generation.log"

if ! "$PYTHON" scripts/generate_prompt.py \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR/prompts" \
    --style-reference assets/examples/sample_caption.txt \
    > "$PROMPT_LOG" 2>&1; then
    echo "WARNING: some prompts failed, see $PROMPT_LOG"
    tail -n 20 "$PROMPT_LOG"
fi

# --------------------------------------------------
# 3. Run ReGenHuman on each video
# --------------------------------------------------

INDEX=0
FAILED=0

while IFS= read -r VIDEO || [[ -n "$VIDEO" ]]; do
    [[ -z "$VIDEO" ]] && continue

    INDEX=$((INDEX + 1))

    NAME="$(basename "$VIDEO")"
    STEM="${NAME%.*}"
    ID="$(printf '%03d_%s' "$INDEX" "$STEM")"

    PROMPT_FILE="$OUTPUT_DIR/prompts/$ID.txt"
    LOG_FILE="$OUTPUT_DIR/logs/$ID.log"
    RAW_OUT="$OUTPUT_DIR/raw/$ID.mp4"
    FINAL_OUT="$OUTPUT_DIR/final/$ID.mp4"

    echo
    echo "=================================================="
    echo "[$INDEX/$COUNT] ReGenHuman: $VIDEO"
    echo "=================================================="

    if [[ ! -s "$PROMPT_FILE" ]]; then
        echo "ERROR: no prompt for $VIDEO (generation failed)"
        FAILED=$((FAILED + 1))
        continue
    fi

    # ---- derive frame settings from the video itself ----
    if ! read -r TOTAL_FRAMES SRC_FPS < <(probe_video "$VIDEO"); then
        echo "ERROR: could not probe $VIDEO with ffprobe"
        FAILED=$((FAILED + 1))
        continue
    fi

    STRIDE="$(awk -v f="$SRC_FPS" -v t="$GEN_FPS" \
        'BEGIN{s=int(f/t+0.5); if(s<1)s=1; print s}')"
    NUM_FRAMES=$(( (TOTAL_FRAMES + STRIDE - 1) / STRIDE ))   # ceil
    FPS_OUT="$(awk -v f="$SRC_FPS" -v s="$STRIDE" 'BEGIN{printf "%.4f", f/s}')"

    echo "Source: $TOTAL_FRAMES frames @ ${SRC_FPS} fps"
    echo "Using : stride=$STRIDE, num_frames=$NUM_FRAMES, fps_out=$FPS_OUT"

    echo "Prompt:"
    cat "$PROMPT_FILE"

    DEMO_ARGS=(
        --input "$VIDEO"
        --variant "$VARIANT"
        --prompt "$(cat "$PROMPT_FILE")"
        --frame_stride "$STRIDE"
        --num_frames "$NUM_FRAMES"
        --fps_out "$FPS_OUT"
        --lora "$LORA"
        --output "$RAW_OUT"
    )
    if [[ -n "$STEPS" ]]; then
        DEMO_ARGS+=(--steps "$STEPS")
    fi

    if ! "$PYTHON" demo.py "${DEMO_ARGS[@]}" > "$LOG_FILE" 2>&1; then
        echo "ERROR: ReGenHuman failed for $VIDEO"
        tail -n 50 "$LOG_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi

    # ---- interpolate back to the source frame rate ----
    if [[ "$INTERPOLATE" == "1" ]]; then
        if ! ffmpeg -y -loglevel error -i "$RAW_OUT" \
            -vf "minterpolate=fps=${SRC_FPS}:mi_mode=mci:mc_mode=aobmc:vsbmc=1" \
            -c:v libx264 -crf 16 -pix_fmt yuv420p "$FINAL_OUT" \
            >> "$LOG_FILE" 2>&1; then
            echo "ERROR: interpolation failed for $VIDEO (raw output kept: $RAW_OUT)"
            FAILED=$((FAILED + 1))
            continue
        fi
    fi

    echo "Successfully processed: $VIDEO"

done < "$MANIFEST"

# --------------------------------------------------
# Summary
# --------------------------------------------------

echo
echo "=================================================="
echo "Batch completed."
echo "Selected videos: $INDEX"
echo "Failures:        $FAILED"
echo "Output directory: $OUTPUT_DIR"
echo "=================================================="

if (( FAILED > 0 )); then
    exit 1
fi