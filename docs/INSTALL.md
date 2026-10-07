# Installation

Verified on Linux, Python 3.12, CUDA 12.4, NVIDIA L40S (46 GB). A single
GPU with ~24 GB VRAM suffices for the demo (the largest stage peaks around
18-22 GB; `--auto_caption` briefly needs ~17 GB on its own).

## 1. Environment

```bash
conda env create -f environment.yml      # creates "regenhuman" (python 3.12 + ffmpeg)
conda activate regenhuman
pip install -e .
```

Or with an existing Python ≥3.10 (3.12 recommended for SAM3):

```bash
pip install -r requirements.txt && pip install -e .
```

Notes:
- `requirements.txt` installs CUDA-enabled `torch==2.6.0` wheels and
  `onnxruntime-gpu`. Both expect a CUDA 12.x driver.
- `ffmpeg` must be on `PATH` (the conda env ships it; otherwise
  `apt install ffmpeg`).

## 2. Conditioning-extraction models

```bash
bash scripts/setup_models.sh
```

Downloads DWPose ONNX checkpoints (~430 MB) and clones Video-Depth-Anything
with its ViT-L weights (~1.5 GB). Add `WITH_VITS=1` to also fetch the
Apache-licensed ViT-S depth weights (see the License section of the README).

## 3. SAM 3 weights (default segmenter — gated)

1. Request access at <https://huggingface.co/facebook/sam3>
2. `huggingface-cli login`

The weights download automatically on first use. Without access, every
command works with `--segmenter grounded_sam2` instead (non-gated
GroundingDINO-tiny + SAM2.1; validated in the paper's segmenter-robustness
ablation).

## 4. Wan2.1-VACE-1.3B base model (~19 GB)

Auto-downloads into `models/` on the first `demo.py` run, from ModelScope by
default. Alternatives:

- Download from Hugging Face instead:
  `export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`
- Already have the weights? Point at them:
  `python demo.py ... --wan_dir /path/to/Wan2.1-VACE-1.3B`
  (the directory must contain `diffusion_pytorch_model.safetensors`,
  `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth`, and `google/umt5-xxl/`).

## 5. Smoke test

```bash
python -c "import torch, onnxruntime, diffsynth, sam3, regenhuman; \
           print('cuda:', torch.cuda.is_available(), \
                 'ort:', onnxruntime.get_available_providers()[0])"
pytest tests/
```

## Disk budget

| Item | Size |
|---|---|
| Conda env | ~12 GB |
| DWPose + Video-Depth-Anything | ~2 GB |
| Wan2.1-VACE-1.3B (`models/`) | ~19 GB |
| SAM3 weights (HF cache) | ~3.5 GB |
| Qwen3-VL-8B (`--auto_caption` only, HF cache) | ~17 GB |
