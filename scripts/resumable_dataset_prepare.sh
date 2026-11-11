#!/bin/bash
# Resumable ReGenHuman dataset preparation. Runs WITHOUT sudo.
# Do any sudo prerequisites yourself first, then run this as your normal user
# (running it via sudo drops the venv and LD_LIBRARY_PATH).
#
# Usage (inside tmux, from anywhere):
#   bash ~/anonymisation-ReGenHuman/run_prepare.sh
#   EVERY=3 bash run_prepare.sh          # use only every 3rd segment
#   VARIANT=structall bash run_prepare.sh
#
# Settings (environment variables, defaults in brackets):
#   VIDEOS_DIR [data/segments]  CAPTIONS [data/captions/captions.csv]
#   VARIANT [structhuman]       OUT [data/dataset_for_regenhuman]
#   SEGMENTER [sam3]            EVERY [1]   MIN_FREE_GB [50]
# Extra arguments are passed on to prepare_resumable.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

VIDEOS_DIR="${VIDEOS_DIR:-data/segments}"
CAPTIONS="${CAPTIONS:-data/captions/captions.csv}"
VARIANT="${VARIANT:-structhuman}"
OUT="${OUT:-data/dataset_for_regenhuman}"
SEGMENTER="${SEGMENTER:-sam3}"
EVERY="${EVERY:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"

if [ "$(id -u)" -eq 0 ]; then
    echo "Do not run this as root/sudo (the venv and LD_LIBRARY_PATH would be lost)."; exit 1
fi

# --- environment (nothing here depends on the state of your shell) ---
source ../.venv/bin/activate
SP="$(python -c "import site; print(site.getsitepackages()[0])")"
NVLIBS="$(find "$SP/nvidia" -type d -name lib 2>/dev/null | paste -sd: - || true)"
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"

# --- preflight: fail early and say which prerequisite is missing ---
fail() { echo "PREFLIGHT FAILED: $1"; exit 1; }

[ -d "$VIDEOS_DIR" ] || fail "videos dir not found: $VIDEOS_DIR (relative to $REPO)"
[ -f "$CAPTIONS" ]   || fail "captions CSV not found: $CAPTIONS"
nvidia-smi -L >/dev/null 2>&1 || fail "no GPU visible (nvidia-smi failed)"

CUDA_PROV="$SP/onnxruntime/capi/libonnxruntime_providers_cuda.so"
[ -f "$CUDA_PROV" ] || fail "onnxruntime-gpu not found in the venv ($CUDA_PROV)"
if ldd "$CUDA_PROV" | grep "not found"; then
    fail "libraries listed above are missing for ONNX Runtime CUDA (cuDNN/cuBLAS/...)"
fi

mkdir -p "$OUT" 2>/dev/null || fail "cannot create $OUT (check ownership, e.g.: sudo chown -R \$USER data)"
[ -w "$OUT" ] || fail "$OUT is not writable (e.g.: sudo chown -R \$USER $OUT)"
NOT_WRITABLE="$(find "$OUT" ! -writable -print -quit 2>/dev/null || true)"
[ -z "$NOT_WRITABLE" ] || fail "not writable (root-owned?): $NOT_WRITABLE  ->  sudo chown -R \$USER $OUT"

FREE_GB="$(df --output=avail -BG "$OUT" | tail -1 | tr -dc '0-9')"
[ "${FREE_GB:-0}" -ge "$MIN_FREE_GB" ] || fail "only ${FREE_GB} GB free on the output disk (MIN_FREE_GB=$MIN_FREE_GB)"

echo "preflight OK | variant=$VARIANT every=$EVERY free=${FREE_GB}GB out=$OUT"

python prepare_resumable.py \
    --videos_dir "$VIDEOS_DIR" \
    --captions "$CAPTIONS" \
    --variant "$VARIANT" \
    --out "$OUT" \
    --segmenter "$SEGMENTER" \
    --num_frames 49 \
    --every "$EVERY" \
    "$@" 2>&1 | tee -a "$OUT/prepare.log"