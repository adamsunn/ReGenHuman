"""Structural-conditioning compositing for the two ReGenHuman variants.

StructAll   : grayscale depth everywhere; inside the human mask the depth is
              soft-blended to the frame's mean depth value; DWPose skeleton
              strokes pasted on top.
StructHuman : original RGB frame; inside the human mask a constant gray
              (128,128,128) fill; DWPose skeleton strokes pasted on top.
"""

import cv2
import numpy as np


class MaskCoverageError(RuntimeError):
    """Raised when human masks are missing/empty and compositing would leak."""


def _paste_pose(base: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Paste DWPose strokes (any pixel with gray value > 10) onto ``base``."""
    gray = cv2.cvtColor(pose, cv2.COLOR_BGR2GRAY)
    _, pm = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    pm_inv = cv2.bitwise_not(pm)
    bg = cv2.bitwise_and(base, base, mask=pm_inv)
    fg = cv2.bitwise_and(pose, pose, mask=pm)
    return cv2.add(bg, fg)


def _blend_fill(base: np.ndarray, mask: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Soft-blend ``fill`` into ``base`` where ``mask`` (uint8 0-255) is set."""
    mf = mask.astype(np.float32) / 255.0
    m3 = np.stack([mf] * 3, axis=-1)
    return (base.astype(np.float32) * (1 - m3) + fill.astype(np.float32) * m3).astype(np.uint8)


def compose_structall(depth: np.ndarray, pose: np.ndarray, human_mask: np.ndarray) -> np.ndarray:
    """StructAll conditioning frame (full-scene regeneration variant)."""
    h, w = depth.shape[:2]
    pose = cv2.resize(pose, (w, h))
    hm = cv2.resize(human_mask, (w, h))
    fill_val = int(np.mean(depth))
    fill = np.full_like(depth, (fill_val, fill_val, fill_val), dtype=np.uint8)
    out = _blend_fill(depth, hm, fill)
    return _paste_pose(out, pose)


def compose_structhuman(frame: np.ndarray, pose: np.ndarray, human_mask: np.ndarray) -> np.ndarray:
    """StructHuman conditioning frame (background-preserving variant)."""
    h, w = frame.shape[:2]
    pose = cv2.resize(pose, (w, h))
    hm = cv2.resize(human_mask, (w, h))
    fill = np.full_like(frame, (128, 128, 128), dtype=np.uint8)
    out = _blend_fill(frame, hm, fill)
    return _paste_pose(out, pose)


def check_mask_coverage(coverages, variant: str, min_coverage: float = 0.0):
    """Validate per-frame human-mask coverage before compositing.

    ``coverages`` is a list of (frame_name, fraction-of-pixels-masked).
    For StructHuman an empty mask means the original person would be pasted
    into the output background verbatim — that is a privacy leak, so we
    hard-fail. For StructAll the person region is already abstracted to
    depth, so we only warn.
    """
    bad = [name for name, c in coverages if c <= min_coverage]
    if not bad:
        return
    msg = (
        f"{len(bad)}/{len(coverages)} frames have no human mask "
        f"(e.g. {bad[:5]}). The segmenter found no person there."
    )
    if variant == "structhuman":
        raise MaskCoverageError(
            msg + " Refusing to composite StructHuman frames: the original "
            "person would appear unmasked in the conditioning video. "
            "Check the clip (is a person visible?) or try --segmenter "
            "grounded_sam2."
        )
    print(f"[warn] {msg} StructAll will show unmasked depth in those frames.")
