#!/usr/bin/env python3
"""Build a ReGenHuman LoRA training dataset from your own videos.

Input : a directory of MP4s and a captions CSV with columns ``video_id,prompt``
        (``video_id`` = file name without extension; dense ~120-160-word
        captions strongly recommended).
Output: for the chosen variant, a dataset directory consumable by
        scripts/train_lora.sh::

    <out>/<variant>/
        videos/{id}.mp4                         ground-truth clip (49 frames)
        vace_videos/{id}_vace.mp4               conditioning clip
        reference_images_conditioning/{id}.jpg  first conditioning frame
        reference_images_original/{id}.jpg      first real frame
        metadata_vace.csv                       video,prompt,vace_video,
                                                vace_reference_image,
                                                vace_reference_image_gt

Usage:
    python scripts/prepare_dataset.py --videos_dir my_videos/ \\
        --captions captions.csv --variant structall --out data/my_dataset \\
        [--segmenter sam3] [--num_frames 49]
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from regenhuman import config, depth, frames, overlay, pose, segment  # noqa: E402


def process_one(video_path, prompt, variant, segmenter, out_root, work_root, num_frames):
    vid = video_path.stem
    work = work_root / vid
    d_frames, d_pose, d_depth, d_masks, d_overlay = (
        work / "frames", work / "pose", work / "depth", work / "masks", work / "overlay")

    src_fps = frames.probe_fps(video_path)
    stride = frames.auto_stride(src_fps)
    n = frames.extract_frames(video_path, d_frames, stride=stride, max_frames=num_frames)
    if n < num_frames:
        print(f"  [skip] {vid}: only {n}/{num_frames} frames at stride {stride}")
        return None
    pose.extract_pose(d_frames, d_pose)
    if variant == "structall":
        depth.extract_depth(d_frames, d_depth)
    coverages = segment.extract_masks(segmenter, d_frames, d_masks)
    try:
        overlay.check_mask_coverage(coverages, variant)
    except overlay.MaskCoverageError as e:
        print(f"  [skip] {vid}: {e}")
        return None

    d_overlay.mkdir(exist_ok=True)
    for frame_path in sorted(d_frames.glob("*.jpg")):
        stem = frame_path.stem
        pose_img = cv2.imread(str(d_pose / f"{stem}.jpg"))
        mask_img = cv2.imread(str(d_masks / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if variant == "structall":
            base = cv2.imread(str(d_depth / f"{stem}.jpg"))
            out = overlay.compose_structall(base, pose_img, mask_img)
        else:
            base = cv2.imread(str(frame_path))
            out = overlay.compose_structhuman(base, pose_img, mask_img)
        cv2.imwrite(str(d_overlay / f"{stem}.jpg"), out)

    fps = src_fps / stride if src_fps > 0 else config.TRAIN_FPS
    (out_root / "videos").mkdir(parents=True, exist_ok=True)
    (out_root / "vace_videos").mkdir(exist_ok=True)
    (out_root / "reference_images_conditioning").mkdir(exist_ok=True)
    (out_root / "reference_images_original").mkdir(exist_ok=True)

    frames.frames_to_video(d_frames, out_root / "videos" / f"{vid}.mp4", fps=fps)
    frames.frames_to_video(d_overlay, out_root / "vace_videos" / f"{vid}_vace.mp4", fps=fps)
    shutil.copy(sorted(d_overlay.glob("*.jpg"))[0],
                out_root / "reference_images_conditioning" / f"{vid}.jpg")
    shutil.copy(sorted(d_frames.glob("*.jpg"))[0],
                out_root / "reference_images_original" / f"{vid}.jpg")
    return {
        "video": f"videos/{vid}.mp4",
        "prompt": prompt,
        "vace_video": f"vace_videos/{vid}_vace.mp4",
        "vace_reference_image": f"reference_images_conditioning/{vid}.jpg",
        "vace_reference_image_gt": f"reference_images_original/{vid}.jpg",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--captions", required=True, help="CSV with columns video_id,prompt")
    ap.add_argument("--variant", required=True, choices=config.VARIANTS)
    ap.add_argument("--out", required=True, help="Output dataset root")
    ap.add_argument("--segmenter", default="sam3", choices=segment.SEGMENTERS)
    ap.add_argument("--num_frames", type=int, default=config.NUM_FRAMES)
    ap.add_argument("--keep_work", action="store_true",
                    help="Keep per-video intermediate frames/pose/depth/masks")
    args = ap.parse_args()

    captions = {}
    with open(args.captions) as f:
        for row in csv.DictReader(f):
            captions[row["video_id"]] = row["prompt"]

    videos = sorted(Path(args.videos_dir).glob("*.mp4"))
    if not videos:
        sys.exit(f"No MP4s in {args.videos_dir}")
    missing = [v.stem for v in videos if v.stem not in captions]
    if missing:
        print(f"[warn] {len(missing)} videos have no caption and are skipped: {missing[:5]}")
    videos = [v for v in videos if v.stem in captions]

    out_root = Path(args.out) / args.variant
    work_root = Path(args.out) / "_work"
    rows = []
    for i, video_path in enumerate(videos):
        print(f"[{i + 1}/{len(videos)}] {video_path.name}")
        row = process_one(video_path, captions[video_path.stem], args.variant,
                          args.segmenter, out_root, work_root, args.num_frames)
        if row:
            rows.append(row)

    csv_path = out_root / "metadata_vace.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video", "prompt", "vace_video", "vace_reference_image", "vace_reference_image_gt"])
        writer.writeheader()
        writer.writerows(rows)
    if not args.keep_work:
        shutil.rmtree(work_root, ignore_errors=True)
    print(f"\n{len(rows)} examples -> {csv_path}")
    print(f"Train with: DATASET_DIR={out_root} OUTPUT_DIR=outputs/{args.variant}_lora "
          f"bash scripts/train_lora.sh")


if __name__ == "__main__":
    main()
