"""Human segmentation with SAM 3 (promptable concept segmentation).

A single "person" text prompt on frame 0 is propagated through the clip; all
detected instances are OR-ed into one binary mask per frame. This is the
segmenter the released LoRAs' training data was built with.

Weights: Hugging Face ``facebook/sam3`` (gated — request access, then
``huggingface-cli login``). If you cannot access them, use the
``grounded_sam2`` fallback segmenter.
"""

import gc
from pathlib import Path

import cv2
import numpy as np

PROMPT = "person"


def _outputs_to_binary_mask(outputs, height, width, torch):
    """OR all SAM3 instance masks of one frame into a single binary mask."""
    combined = np.zeros((height, width), dtype=bool)
    if outputs is None:
        return combined
    masks = outputs.get("out_binary_masks")
    if masks is None or len(masks) == 0:
        return combined
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.shape != (height, width):
            m = cv2.resize(
                m.astype(np.float32), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
        combined = np.logical_or(combined, m.astype(bool))
    return combined


def extract_masks(frames_dir: Path, out_dir: Path):
    """Segment every person in ``frames_dir`` (``*.jpg``) into ``out_dir`` (``*.png``).

    Returns a list of ``(frame_stem, coverage_fraction)`` per frame.
    """
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No JPG frames in {frames_dir}")

    first = cv2.imread(str(frame_files[0]))
    h, w = first.shape[:2]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    predictor = build_sam3_video_predictor()
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(frames_dir))
    )
    session_id = response["session_id"]

    response = predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=PROMPT)
    )
    initial = response.get("outputs")
    obj_ids = initial.get("out_obj_ids") if initial is not None else None
    n_detected = len(obj_ids) if obj_ids is not None else 0

    masks = {i: np.zeros((h, w), dtype=bool) for i in range(len(frame_files))}
    if n_detected > 0:
        for resp in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            masks[resp["frame_index"]] = _outputs_to_binary_mask(
                resp["outputs"], h, w, torch
            )

    coverages = []
    for i, frame_path in enumerate(frame_files):
        hm = np.asarray(masks[i]).squeeze()
        cv2.imwrite(str(out_dir / f"{frame_path.stem}.png"), (hm * 255).astype(np.uint8))
        coverages.append((frame_path.stem, float(hm.mean())))

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    del predictor, masks
    gc.collect()
    torch.cuda.empty_cache()
    return coverages
