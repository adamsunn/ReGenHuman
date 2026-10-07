"""Video-Depth-Anything grayscale depth extraction.

Produces one grayscale depth JPG per frame (per-frame min-max normalized to
0-255, matching the LoRA training data). The Video-Depth-Anything repository
is cloned by ``scripts/setup_models.sh`` into ``third_party/`` and imported
from there.

License note: the ViT-Large weights are CC-BY-NC-4.0 (non-commercial); the
ViT-Small weights are Apache-2.0. The released LoRAs were trained on ViT-L
depth maps — using ``vits`` is possible but slightly off-distribution.
"""

import gc
import sys
from pathlib import Path

import cv2
import numpy as np

from . import config

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def extract_depth(
    frames_dir: Path,
    out_dir: Path,
    encoder: str = "vitl",
    checkpoint: Path = None,
    input_size: int = 518,
    max_res: int = 1280,
    fp32: bool = False,
) -> int:
    """Estimate depth for every ``*.jpg`` in ``frames_dir``; frees the model on exit."""
    import torch

    vda_repo = config.VDA_REPO_DIR
    if not vda_repo.exists():
        raise FileNotFoundError(
            f"Video-Depth-Anything not found at {vda_repo}. Run scripts/setup_models.sh."
        )
    if str(vda_repo) not in sys.path:
        sys.path.insert(0, str(vda_repo))
    from video_depth_anything.video_depth import VideoDepthAnything

    checkpoint = Path(checkpoint or config.VDA_CHECKPOINTS[encoder])
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Depth checkpoint missing: {checkpoint}. Run scripts/setup_models.sh."
        )

    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No JPG frames in {frames_dir}")

    frames = []
    shape = None
    for frame_path in frame_files:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Unreadable frame: {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        if max(h, w) > max_res:
            scale = max_res / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        if shape is None:
            shape = frame.shape[:2]
        elif frame.shape[:2] != shape:
            raise RuntimeError(
                f"Frame shape mismatch in {frames_dir}: {frame.shape[:2]} vs {shape} "
                "(all frames of one video must share a resolution)"
            )
        frames.append(frame)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoDepthAnything(**_MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"), strict=True)
    model = model.to(device).eval()

    # The fps argument only matters for VDA's streaming mode; pass the
    # training-time sampling rate for consistency.
    depths, _ = model.infer_video_depth(
        np.array(frames), config.TRAIN_FPS,
        input_size=input_size, device=device, fp32=fp32,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, depth in enumerate(depths):
        # Per-frame min-max normalization to 0-255 (matches training data).
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        depth_gray = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(out_dir / f"{i + 1:06d}.jpg"), depth_gray)

    n = len(depths)
    del model, depths, frames
    gc.collect()
    torch.cuda.empty_cache()
    return n
