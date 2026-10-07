"""Central defaults for the ReGenHuman pipeline.

These mirror the configuration the released LoRAs were trained and evaluated
with; change them only if you know what you are doing.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- variants
# Public name -> conditioning semantics (see regenhuman/overlay.py).
VARIANTS = ("structall", "structhuman")

# ---------------------------------------------------------------- geometry
WIDTH = 832
HEIGHT = 480
NUM_FRAMES = 49          # Wan2.1-VACE context window used in training
TRAIN_FPS = 7.5          # training clips were sampled at ~7.5 fps
OUTPUT_FPS = 15

# ---------------------------------------------------------------- sampling
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 5.0
SEED = 1
CHUNK_SIZE = 49          # for videos longer than NUM_FRAMES
FRAME_OVERLAP = 5

# Standard Wan negative prompt (from the Wan2.1 reference inference code).
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# ---------------------------------------------------------------- weights
WEIGHTS_DIR = REPO_ROOT / "weights"
DEFAULT_LORA_STEP = 16000
LORA_ALPHA = 1.0

def default_lora_path(variant: str, step: int = DEFAULT_LORA_STEP) -> Path:
    return WEIGHTS_DIR / variant / f"step-{step}.safetensors"

# Downloaded third-party models (populated by scripts/setup_models.sh).
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
DWPOSE_DET_ONNX = THIRD_PARTY_DIR / "dwpose_ckpts" / "yolox_l.onnx"
DWPOSE_POSE_ONNX = THIRD_PARTY_DIR / "dwpose_ckpts" / "dw-ll_ucoco_384.onnx"
VDA_REPO_DIR = THIRD_PARTY_DIR / "Video-Depth-Anything"
VDA_CHECKPOINTS = {
    "vitl": VDA_REPO_DIR / "checkpoints" / "video_depth_anything_vitl.pth",
    "vits": VDA_REPO_DIR / "checkpoints" / "video_depth_anything_vits.pth",
}

# Wan2.1-VACE base weights auto-download into ./models/ (ModelScope layout);
# pass --wan_dir to demo.py to use a pre-downloaded copy instead.
WAN_MODELS_DIR = REPO_ROOT / "models"

# ---------------------------------------------------------------- caption
CAPTION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
