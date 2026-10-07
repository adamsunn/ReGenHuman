"""Fallback human segmenter: GroundingDINO-tiny + SAM 2.1 (hiera-tiny).

GroundingDINO detects "person" boxes on frame 0; the SAM2 video predictor
propagates each box through the clip; all person masks are OR-ed per frame.
None of the weights are gated, so this path works without SAM3 access. It is
the "weaker segmenter" from the paper's robustness ablation — privacy and
quality were shown to be nearly unaffected by this swap.
"""

import gc
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .. import config

SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # resolved inside the sam2 package
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "person."
BOX_THRESHOLD = 0.25


def _sam2_checkpoint() -> Path:
    ckpt = config.THIRD_PARTY_DIR / "sam2_ckpts" / "sam2.1_hiera_tiny.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SAM2.1 hiera-tiny checkpoint -> {ckpt}")
        urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, ckpt)
    return ckpt


def extract_masks(frames_dir: Path, out_dir: Path):
    """Segment persons in ``frames_dir`` (``*.jpg``) into ``out_dir`` (``*.png``).

    Returns a list of ``(frame_stem, coverage_fraction)`` per frame.
    """
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No JPG frames in {frames_dir}")

    first = cv2.imread(str(frame_files[0]))
    src_h, src_w = first.shape[:2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    video_predictor = build_sam2_video_predictor(SAM2_CONFIG, str(_sam2_checkpoint()))
    processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL_ID
    ).to(device)

    # Detect person boxes on frame 0.
    img0 = Image.open(frame_files[0]).convert("RGB")
    inputs = processor(images=img0, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, target_sizes=[img0.size[::-1]],
    )
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    boxes = boxes[scores >= BOX_THRESHOLD]

    video_segments = {}
    if len(boxes) > 0:
        inference_state = video_predictor.init_state(video_path=str(frames_dir))
        for obj_id, box in enumerate(boxes, start=1):
            video_predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=0, obj_id=obj_id, box=box,
            )
        for out_frame_idx, out_obj_ids, out_mask_logits in \
                video_predictor.propagate_in_video(inference_state):
            combined = None
            for i in range(len(out_obj_ids)):
                m = (out_mask_logits[i] > 0.0).cpu().numpy()
                m = np.squeeze(m)
                if m.ndim > 2:
                    m = m[0]
                combined = m if combined is None else np.logical_or(combined, m)
            video_segments[out_frame_idx] = combined
        del inference_state

    coverages = []
    for frame_idx, frame_path in enumerate(frame_files):
        mask = video_segments.get(frame_idx)
        if mask is None:
            mask = np.zeros((src_h, src_w), dtype=bool)
        mask = np.squeeze(mask)
        if mask.shape != (src_h, src_w):
            mask = cv2.resize(
                mask.astype(np.uint8), (src_w, src_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        cv2.imwrite(str(out_dir / f"{frame_path.stem}.png"), (mask * 255).astype(np.uint8))
        coverages.append((frame_path.stem, float(mask.mean())))

    del video_predictor, processor, grounding_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return coverages
