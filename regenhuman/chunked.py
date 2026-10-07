"""
Chunked-generation helpers for the Wan2.1-VACE pipeline.

The Wan VAE is 4:1 temporal, so every chunk must satisfy
``num_frames % 4 == 1``. We always pad the last frame as needed and trim
the output back to the requested length, so callers can pass any
non-empty list of control frames.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image


def _solid_frame(size, value: int):
    """A solid RGB PIL frame of ``size`` (W, H) — used for VACE masks."""
    return Image.new("RGB", size, (value, value, value))


def generate_segment(
    pipe,
    control_frames: List,
    ref_img,
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_inference_steps: int = 50,
    cfg_scale: float = 5.0,
    mask_frames: Optional[List] = None,
):
    """Generate a single clip whose length matches ``len(control_frames)``.

    Pads the conditioning by repeating the last frame to satisfy
    ``num_frames % 4 == 1``; trims the generated output back to the
    original length.

    ``mask_frames`` (optional) is a per-frame VACE inpainting mask, same length
    as ``control_frames``: a white frame (value 1) marks a frame as *reactive*
    (generated from the control), a black frame (0) marks it *inactive* — the
    corresponding ``control_frames`` entry is kept as known ground-truth RGB.
    This is how we carry previously-generated frames into the next chunk.
    """
    original_n = len(control_frames)
    gen_n = original_n
    ctrl = list(control_frames)
    mask = list(mask_frames) if mask_frames is not None else None
    if gen_n % 4 != 1:
        gen_n = ((original_n - 1) // 4 + 1) * 4 + 1
        pad = gen_n - original_n
        ctrl = ctrl + [ctrl[-1]] * pad
        if mask is not None:
            # Padded tail is trimmed away; mark it reactive (white) for safety.
            mask = mask + [_solid_frame(ctrl[0].size, 255)] * pad

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        vace_video=ctrl,
        num_frames=gen_n,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        seed=seed,
        tiled=True,
    )
    if mask is not None:
        kwargs["vace_video_mask"] = mask
    if ref_img is not None:
        kwargs["vace_reference_image"] = ref_img

    video = pipe(**kwargs)
    if len(video) > original_n:
        video = video[:original_n]
    return list(video)


def generate_chunked(
    pipe,
    control_frames_full: List,
    ref_img,
    prompt: str,
    negative_prompt: str,
    seed: int,
    chunk_size: int,
    frame_overlap: int,
    num_inference_steps: int = 50,
    cfg_scale: float = 5.0,
):
    """Generate a long video in overlapping chunks with true temporal continuity.

    A fixed reference image anchors identity across chunks. For every chunk
    after the first, the ``frame_overlap`` frames it shares with the previous
    chunk are injected as **known ground-truth RGB** (the pixels we already
    generated) via a VACE inpainting mask — those frames are marked *inactive*
    (mask 0) so the model continues from them instead of re-hallucinating them,
    while the remaining frames are generated from the overlay control (mask 1).
    We then drop the reproduced overlap prefix during concatenation.

    ``all_frames`` is kept aligned to the global control index, so the known
    frames for chunk ``i`` are exactly ``all_frames[start : start+overlap]``.
    """
    if chunk_size % 4 != 1:
        raise ValueError(
            f"chunk_size must satisfy chunk_size % 4 == 1; got {chunk_size}"
        )
    if frame_overlap < 0 or frame_overlap >= chunk_size:
        raise ValueError(
            f"frame_overlap must be in [0, {chunk_size}); got {frame_overlap}"
        )

    total = len(control_frames_full)
    stride = chunk_size - frame_overlap
    num_chunks = max(1, (total - frame_overlap + stride - 1) // stride)
    size = control_frames_full[0].size  # (W, H)
    white = _solid_frame(size, 255)      # reactive → generate
    black = _solid_frame(size, 0)        # inactive → keep known RGB
    print(
        f"[Chunked] total={total}, chunk_size={chunk_size}, overlap={frame_overlap}, "
        f"stride={stride}, num_chunks={num_chunks}"
    )

    all_frames = []
    for i in range(num_chunks):
        start = i * stride
        end = min(start + chunk_size, total)
        win = end - start
        chunk_ctrl = [control_frames_full[j] for j in range(start, end)]
        mask_frames = None
        if i > 0:
            # First `frame_overlap` frames: previously generated RGB, kept as
            # known content; the rest generated from the overlay control.
            known = all_frames[start:start + frame_overlap]
            k = len(known)
            chunk_ctrl = list(known) + chunk_ctrl[k:]
            mask_frames = [black] * k + [white] * (win - k)
        print(
            f"[Chunked] Chunk {i + 1}/{num_chunks}: frames [{start}, {end}) "
            f"({win} frames" + (f", {frame_overlap} carried-over)" if i > 0 else ")")
        )
        # Consistent seed across chunks: continuity now comes from the injected
        # known frames, not from noise-pattern juggling.
        chunk_frames = generate_segment(
            pipe=pipe,
            control_frames=chunk_ctrl,
            ref_img=ref_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            mask_frames=mask_frames,
        )
        if i == 0:
            all_frames.extend(chunk_frames)
        else:
            all_frames.extend(chunk_frames[frame_overlap:])
        torch.cuda.empty_cache()

    print(f"[Chunked] Stitched output: {len(all_frames)} frames (expected {total})")
    return all_frames


def generate(
    pipe,
    control_frames: List,
    ref_img,
    prompt: str,
    negative_prompt: str,
    seed: int,
    chunk_size: int = 49,
    frame_overlap: int = 5,
    enable_chunked: bool = True,
    num_inference_steps: int = 50,
    cfg_scale: float = 5.0,
):
    """Single entry point: route between single-shot and chunked generation."""
    if enable_chunked and len(control_frames) > chunk_size:
        return generate_chunked(
            pipe=pipe,
            control_frames_full=control_frames,
            ref_img=ref_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            chunk_size=chunk_size,
            frame_overlap=frame_overlap,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
        )
    return generate_segment(
        pipe=pipe,
        control_frames=control_frames,
        ref_img=ref_img,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
    )
