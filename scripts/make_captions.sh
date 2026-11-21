#!/bin/bash
# Caption every segment with the Qwen3-VL script and merge the results into one CSV.
#
# Usage:
#   CAPTION_SCRIPT=caption_videos.py [STYLE_REF=style.txt] \
#     bash make_captions.sh data/segments data/captions
#
# Result: data/captions/captions.csv  (columns video_id,prompt)
set -euo pipefail

SEG_DIR="${1:?segments directory}"
OUT_DIR="${2:?output directory}"
CAPTION_SCRIPT="${CAPTION_SCRIPT:-caption_videos.py}"   # <- your captioning script
STYLE_REF="${STYLE_REF:-}"
MAX_FPS="${MAX_FPS:-8}"          # segments are only ~2-4 s long, so sample denser than the default 2
MAX_FRAMES="${MAX_FRAMES:-64}"

mkdir -p "$OUT_DIR/txt"
find "$SEG_DIR" -maxdepth 1 -name '*.mp4' | sort > "$OUT_DIR/manifest.txt"
echo "$(wc -l < "$OUT_DIR/manifest.txt") segments in manifest"

extra=()
[ -n "$STYLE_REF" ] && extra+=(--style-reference "$STYLE_REF")

# The captioner exits 1 if any single video failed; still merge what succeeded.
python "$CAPTION_SCRIPT" \
    --manifest "$OUT_DIR/manifest.txt" \
    --output-dir "$OUT_DIR/txt" \
    --max-fps "$MAX_FPS" --max-frames "$MAX_FRAMES" \
    ${extra[@]+"${extra[@]}"} || echo "[warn] some videos failed to caption"

python "$(dirname "$0")/captions_to_csv.py" \
    --captions_dir "$OUT_DIR/txt" \
    --videos_dir "$SEG_DIR" \
    --out "$OUT_DIR/captions.csv"