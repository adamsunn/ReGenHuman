"""Unit tests for the conditioning compositor and the privacy leak guard."""

import numpy as np
import pytest

from regenhuman.overlay import (
    MaskCoverageError,
    check_mask_coverage,
    compose_structall,
    compose_structhuman,
)

H, W = 64, 96


def _fixtures():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    depth = np.repeat(rng.integers(0, 256, (H, W, 1), dtype=np.uint8), 3, axis=2)
    pose = np.zeros((H, W, 3), dtype=np.uint8)
    pose[10:12, :, :] = (0, 0, 255)  # a horizontal red "limb"
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[20:50, 30:70] = 255
    return frame, depth, pose, mask


def test_structall_fills_mask_with_mean_depth():
    _, depth, pose, mask = _fixtures()
    out = compose_structall(depth, pose, mask)
    fill = int(np.mean(depth))
    # Inside the mask (away from pose strokes) = mean-depth gray.
    assert np.all(out[30:40, 40:60] == fill)
    # Outside the mask = untouched depth.
    assert np.array_equal(out[55:60, 0:20], depth[55:60, 0:20])
    # Pose strokes survive on top.
    assert np.all(out[10, :, 2] == 255)


def test_structhuman_fills_mask_with_gray_128():
    frame, _, pose, mask = _fixtures()
    out = compose_structhuman(frame, pose, mask)
    assert np.all(out[30:40, 40:60] == 128)
    assert np.array_equal(out[55:60, 0:20], frame[55:60, 0:20])
    assert np.all(out[10, :, 2] == 255)


def test_soft_blend_on_mask_edge():
    """A 50% mask value must blend, not binarize (matches training data)."""
    frame, _, pose, mask = _fixtures()
    mask = mask.copy()
    mask[20:50, 30:70] = 127  # half-strength mask
    out = compose_structhuman(frame, pose, mask)
    inside = out[30:40, 40:60].astype(np.float32)
    expected = frame[30:40, 40:60].astype(np.float32) * (1 - 127 / 255) + 128.0 * (127 / 255)
    assert np.max(np.abs(inside - expected)) <= 1.0


def test_leak_guard_hard_fails_structhuman():
    coverages = [("000001", 0.2), ("000002", 0.0)]
    with pytest.raises(MaskCoverageError):
        check_mask_coverage(coverages, "structhuman")


def test_leak_guard_warns_but_passes_structall():
    coverages = [("000001", 0.2), ("000002", 0.0)]
    check_mask_coverage(coverages, "structall")  # must not raise


def test_leak_guard_passes_with_full_coverage():
    coverages = [("000001", 0.2), ("000002", 0.3)]
    check_mask_coverage(coverages, "structhuman")
