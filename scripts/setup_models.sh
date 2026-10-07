#!/bin/bash
# Download the conditioning-extraction models into third_party/.
# (The Wan2.1-VACE-1.3B base model auto-downloads on the first demo run.)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$REPO_ROOT/third_party"
mkdir -p "$TP"

echo "== DWPose ONNX checkpoints (Apache-2.0, HF yzd-v/DWPose) =="
mkdir -p "$TP/dwpose_ckpts"
for f in yolox_l.onnx dw-ll_ucoco_384.onnx; do
    if [ ! -f "$TP/dwpose_ckpts/$f" ]; then
        curl -L --fail -o "$TP/dwpose_ckpts/$f" \
            "https://huggingface.co/yzd-v/DWPose/resolve/main/$f"
    else
        echo "  $f already present"
    fi
done

echo "== Video-Depth-Anything (code Apache-2.0) =="
VDA="$TP/Video-Depth-Anything"
VDA_COMMIT=859cce40be80afe2cdd44806466fe154f09bee2a
if [ ! -d "$VDA" ]; then
    git clone https://github.com/DepthAnything/Video-Depth-Anything "$VDA"
fi
git -C "$VDA" checkout -q "$VDA_COMMIT"
mkdir -p "$VDA/checkpoints"
# Default (matches the released LoRAs' training data): ViT-Large.
# NOTE: the ViT-L weights are CC-BY-NC-4.0 (non-commercial). The ViT-S
# weights are Apache-2.0; pass --depth_encoder vits to demo.py to use them.
if [ ! -f "$VDA/checkpoints/video_depth_anything_vitl.pth" ]; then
    curl -L --fail -o "$VDA/checkpoints/video_depth_anything_vitl.pth" \
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth"
fi
if [ "${WITH_VITS:-0}" = "1" ] && [ ! -f "$VDA/checkpoints/video_depth_anything_vits.pth" ]; then
    curl -L --fail -o "$VDA/checkpoints/video_depth_anything_vits.pth" \
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
fi

cat <<'EOF'

== SAM 3 (default segmenter) ==
SAM3 weights are GATED on Hugging Face (facebook/sam3):
  1. Request access at https://huggingface.co/facebook/sam3
  2. huggingface-cli login
The weights then download automatically on first use.
No access? Use `demo.py --segmenter grounded_sam2` (non-gated fallback;
its SAM2.1 checkpoint downloads automatically on first use).

Setup complete.
EOF
