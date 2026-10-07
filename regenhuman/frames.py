"""Frame sampling and video encoding utilities."""

import os
import subprocess
from pathlib import Path

import cv2

from . import config


def probe_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps


def auto_stride(src_fps: float, target_fps: float = config.TRAIN_FPS) -> int:
    """Pick a frame stride so sampled frames land near the training fps.

    The LoRAs were trained on clips sampled at ~7.5 fps (every 4th frame of
    ~30 fps video); matching that temporal density at inference matters.
    """
    if src_fps <= 0:
        return 4
    return max(1, round(src_fps / target_fps))


def extract_frames(
    video_path: Path,
    output_dir: Path,
    stride: int = 1,
    max_frames: int = None,
) -> int:
    """Save every ``stride``-th frame as 1-based ``%06d.jpg``.

    The 1-based naming is shared by every pipeline stage (pose/depth/mask
    files are matched by stem), so keep it consistent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    read_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if read_count % stride == 0:
            saved_count += 1
            cv2.imwrite(str(output_dir / f"{saved_count:06d}.jpg"), frame)
            if max_frames is not None and saved_count >= max_frames:
                break
        read_count += 1
    cap.release()
    return saved_count


def _ffmpeg_bin() -> str:
    """Prefer the static ffmpeg bundled with imageio-ffmpeg (always has
    libx264); system ffmpeg builds frequently lack it. Override with the
    ``FFMPEG`` env var."""
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def frames_to_video(frames_dir: Path, output_video: Path, fps: float):
    """Encode a directory of ``%06d.jpg`` frames to H.264 MP4 via ffmpeg.

    Paths are prefixed with ``file:`` so names containing ``:`` are not
    parsed as protocol prefixes by ffmpeg.
    """
    ffmpeg_bin = _ffmpeg_bin()
    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", f"file:{Path(frames_dir) / '%06d.jpg'}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-b:v", "2M",
        f"file:{output_video}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}).\nCMD: {' '.join(cmd)}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
