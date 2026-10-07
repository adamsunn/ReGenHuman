"""DWPose whole-body skeleton rendering (ONNX, GPU via onnxruntime).

Produces one skeleton-on-black JPG per input frame, matching the rendering
used to build the LoRA training data. Unlike the research script this runs
in-process, takes explicit checkpoint paths (no chdir tricks), and runs
detection + pose estimation once per frame.
"""

from pathlib import Path

import cv2
from tqdm import tqdm

from . import config
from .dwpose_onnx import DWposeDetector


def extract_pose(
    frames_dir: Path,
    out_dir: Path,
    onnx_det: Path = None,
    onnx_pose: Path = None,
) -> int:
    """Render DWPose skeletons for every ``*.jpg`` in ``frames_dir``.

    Returns the number of frames processed.
    """
    onnx_det = Path(onnx_det or config.DWPOSE_DET_ONNX)
    onnx_pose = Path(onnx_pose or config.DWPOSE_POSE_ONNX)
    for p in (onnx_det, onnx_pose):
        if not p.exists():
            raise FileNotFoundError(
                f"DWPose checkpoint missing: {p}\nRun scripts/setup_models.sh first."
            )

    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No JPG frames in {frames_dir}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = DWposeDetector(onnx_det, onnx_pose)
    n = 0
    for frame_path in tqdm(frame_files, desc="pose", unit="frame"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[warn] unreadable frame skipped: {frame_path}")
            continue
        pose_vis = detector(frame)
        cv2.imwrite(str(out_dir / f"{frame_path.stem}.jpg"), pose_vis)
        n += 1

    # Free the ONNX runtime sessions (GPU memory) before the next stage.
    del detector
    return n
