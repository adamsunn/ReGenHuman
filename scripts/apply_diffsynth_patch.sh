#!/bin/bash
# Clone DiffSynth-Studio at the pinned commit and apply the ReGenHuman
# training patch. ONLY TRAINING needs the patched clone — inference (demo.py)
# runs on the unpatched pip-installed diffsynth.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFSYNTH_DIR="${1:-$REPO_ROOT/third_party/DiffSynth-Studio}"
COMMIT=a6884f6

if [ ! -d "$DIFFSYNTH_DIR" ]; then
    git clone https://github.com/modelscope/DiffSynth-Studio "$DIFFSYNTH_DIR"
fi
git -C "$DIFFSYNTH_DIR" checkout -q "$COMMIT"
git -C "$DIFFSYNTH_DIR" apply --check "$REPO_ROOT/patches/diffsynth_training_a6884f6.patch"
git -C "$DIFFSYNTH_DIR" apply "$REPO_ROOT/patches/diffsynth_training_a6884f6.patch"
echo "DiffSynth-Studio @ $COMMIT patched at: $DIFFSYNTH_DIR"
echo "Train with: bash scripts/train_lora.sh (see docs/TRAINING.md)"
