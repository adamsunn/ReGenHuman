"""Human-mask segmenters. ``sam3`` is the default (matches the training data);
``grounded_sam2`` is a non-gated fallback validated in the paper's segmenter
ablation."""

SEGMENTERS = ("sam3", "grounded_sam2")


def extract_masks(name: str, frames_dir, out_dir):
    """Run the chosen segmenter; returns [(frame_stem, coverage_fraction)]."""
    if name == "sam3":
        from .sam3_person import extract_masks as fn
    elif name == "grounded_sam2":
        from .grounded_sam2 import extract_masks as fn
    else:
        raise ValueError(f"Unknown segmenter {name!r}; choose from {SEGMENTERS}")
    return fn(frames_dir, out_dir)
