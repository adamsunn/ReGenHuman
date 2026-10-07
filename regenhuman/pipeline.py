"""End-to-end ReGenHuman anonymization pipeline.

Stages (each loads its model, runs, and frees GPU memory before the next):
  1. frames   — sample input video to JPG frames at ~7.5 fps
  2. pose     — DWPose skeleton rendering (ONNX)
  3. depth    — Video-Depth-Anything grayscale depth
  4. masks    — human segmentation (SAM3 or GroundingDINO+SAM2)
  5. overlay  — StructAll / StructHuman conditioning compositing
  6. caption  — optional Qwen3-VL dense captioning
  7. generate — Wan2.1-VACE-1.3B + ReGenHuman LoRA

With ``resume=True``, stages whose outputs already exist in the work dir are
skipped, so a failed run can be re-entered cheaply.
"""

import hashlib
import json
import platform
import time
from pathlib import Path

import cv2

from . import config, frames, overlay, pose, depth, segment, generate


def _count(d: Path, pattern: str) -> int:
    return len(list(d.glob(pattern))) if d.exists() else 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(
    input_video: Path,
    variant: str,
    work_dir: Path,
    prompt: str = None,
    auto_caption: bool = False,
    segmenter: str = "sam3",
    lora_path: Path = None,
    output_path: Path = None,
    wan_dir: Path = None,
    num_frames: int = None,
    frame_stride: int = None,
    width: int = config.WIDTH,
    height: int = config.HEIGHT,
    steps: int = config.NUM_INFERENCE_STEPS,
    cfg_scale: float = config.CFG_SCALE,
    seed: int = config.SEED,
    fps_out: float = config.OUTPUT_FPS,
    chunk_size: int = config.CHUNK_SIZE,
    frame_overlap: int = config.FRAME_OVERLAP,
    depth_encoder: str = "vitl",
    resume: bool = False,
) -> Path:
    from PIL import Image

    if variant not in config.VARIANTS:
        raise ValueError(f"variant must be one of {config.VARIANTS}")
    if not prompt and not auto_caption:
        raise ValueError(
            "Provide --prompt (a dense scene/action description) or pass "
            "--auto_caption. Generation quality degrades badly with no prompt."
        )

    input_video = Path(input_video)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    d_frames = work_dir / "frames"
    d_pose = work_dir / "pose"
    d_depth = work_dir / "depth"
    d_masks = work_dir / "masks"
    d_overlay = work_dir / "overlay"
    output_path = Path(output_path or work_dir / "output.mp4")
    timings = {}

    # ---- 1. frame sampling -------------------------------------------------
    src_fps = frames.probe_fps(input_video)
    stride = frame_stride or frames.auto_stride(src_fps)
    t0 = time.time()
    if resume and _count(d_frames, "*.jpg") > 0:
        n_frames = _count(d_frames, "*.jpg")
        print(f"[1/7] frames: reusing {n_frames} existing")
    else:
        n_frames = frames.extract_frames(input_video, d_frames, stride=stride, max_frames=num_frames)
        print(f"[1/7] frames: {n_frames} sampled (src {src_fps:.1f} fps, stride {stride})")
    if n_frames == 0:
        raise RuntimeError(f"No frames decoded from {input_video}")
    timings["frames"] = time.time() - t0

    # ---- 2. pose -----------------------------------------------------------
    t0 = time.time()
    if resume and _count(d_pose, "*.jpg") >= n_frames:
        print("[2/7] pose: reusing existing")
    else:
        pose.extract_pose(d_frames, d_pose)
        print(f"[2/7] pose: {_count(d_pose, '*.jpg')} skeleton frames")
    timings["pose"] = time.time() - t0

    # ---- 3. depth (StructAll only — StructHuman composites over RGB) -------
    t0 = time.time()
    if variant != "structall":
        print("[3/7] depth: skipped (not needed for structhuman)")
    elif resume and _count(d_depth, "*.jpg") >= n_frames:
        print("[3/7] depth: reusing existing")
    else:
        depth.extract_depth(d_frames, d_depth, encoder=depth_encoder)
        print(f"[3/7] depth: {_count(d_depth, '*.jpg')} depth maps ({depth_encoder})")
    timings["depth"] = time.time() - t0

    # ---- 4. human masks ----------------------------------------------------
    t0 = time.time()
    if resume and _count(d_masks, "*.png") >= n_frames:
        coverages = []
        for p in sorted(d_masks.glob("*.png")):
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            coverages.append((p.stem, float((m > 127).mean()) if m is not None else 0.0))
        print("[4/7] masks: reusing existing")
    else:
        coverages = segment.extract_masks(segmenter, d_frames, d_masks)
        with_person = sum(1 for _, c in coverages if c > 0)
        print(f"[4/7] masks: person found in {with_person}/{len(coverages)} frames ({segmenter})")
    overlay.check_mask_coverage(coverages, variant)
    timings["masks"] = time.time() - t0

    # ---- 5. overlay compositing -------------------------------------------
    t0 = time.time()
    d_overlay.mkdir(exist_ok=True)
    for frame_path in sorted(d_frames.glob("*.jpg")):
        stem = frame_path.stem
        pose_img = cv2.imread(str(d_pose / f"{stem}.jpg"))
        mask_img = cv2.imread(str(d_masks / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if pose_img is None or mask_img is None:
            raise RuntimeError(f"Missing pose or mask for frame {stem}")
        if variant == "structall":
            depth_img = cv2.imread(str(d_depth / f"{stem}.jpg"))
            if depth_img is None:
                raise RuntimeError(f"Missing depth for frame {stem}")
            out = overlay.compose_structall(depth_img, pose_img, mask_img)
        else:
            frame_img = cv2.imread(str(frame_path))
            out = overlay.compose_structhuman(frame_img, pose_img, mask_img)
        cv2.imwrite(str(d_overlay / f"{stem}.jpg"), out)
    cond_fps = src_fps / stride if src_fps > 0 else config.TRAIN_FPS
    frames.frames_to_video(d_overlay, work_dir / "conditioning.mp4", fps=cond_fps)
    print(f"[5/7] overlay: {_count(d_overlay, '*.jpg')} {variant} conditioning frames")
    timings["overlay"] = time.time() - t0

    # ---- 6. caption --------------------------------------------------------
    t0 = time.time()
    caption_file = work_dir / "caption.txt"
    if not prompt:
        if resume and caption_file.exists():
            prompt = caption_file.read_text().strip()
            print("[6/7] caption: reusing existing")
        else:
            from . import caption as caption_mod
            prompt = caption_mod.caption_video(d_frames)
            print(f"[6/7] caption ({len(prompt)} chars): {prompt[:120]}...")
    else:
        print("[6/7] caption: using provided --prompt")
    caption_file.write_text(prompt + "\n")
    timings["caption"] = time.time() - t0

    # ---- 7. generation -----------------------------------------------------
    t0 = time.time()
    overlay_files = sorted(d_overlay.glob("*.jpg"))
    overlay_frames = [
        Image.open(p).convert("RGB").resize((width, height)) for p in overlay_files
    ]
    # Reference image = first overlay frame (how the LoRAs were trained).
    ref_img = overlay_frames[0]
    ref_img.save(work_dir / "ref.png")
    lora_path = Path(lora_path or config.default_lora_path(variant))
    print(f"[7/7] generate: {len(overlay_frames)} frames, LoRA {lora_path.name}, "
          f"{steps} steps, cfg {cfg_scale}, seed {seed}")
    generate.generate_video(
        overlay_frames, ref_img, prompt, lora_path, output_path,
        wan_dir=wan_dir, num_inference_steps=steps, cfg_scale=cfg_scale,
        seed=seed, chunk_size=chunk_size, frame_overlap=frame_overlap, fps=fps_out,
    )
    timings["generate"] = time.time() - t0

    # ---- provenance --------------------------------------------------------
    meta = {
        "input": str(input_video),
        "variant": variant,
        "segmenter": segmenter,
        "depth_encoder": depth_encoder,
        "prompt": prompt,
        "lora": {"path": str(lora_path), "sha256": _sha256(lora_path)},
        "params": {
            "width": width, "height": height, "num_frames": len(overlay_frames),
            "frame_stride": stride, "steps": steps, "cfg_scale": cfg_scale,
            "seed": seed, "fps_out": fps_out,
        },
        "timings_sec": {k: round(v, 1) for k, v in timings.items()},
        "host": platform.node(),
    }
    try:
        import torch
        meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except Exception:
        pass
    (work_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nDone. Output: {output_path}")
    return output_path
