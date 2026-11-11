#!/usr/bin/env python3
"""Cut long clips into non-overlapping segments for scripts/prepare_dataset.py.

prepare_dataset.py takes only the FIRST --num_frames (49) frames of a video,
sampled with stride = frames.auto_stride(fps). So every training segment must
contain exactly num_frames * stride source frames; otherwise it is skipped
("only n/49 frames") or most of the clip is thrown away.

Find the stride first (from the ReGenHuman repo root):
    python -c "from regenhuman import frames; print(frames.auto_stride(25.0))"
(use your clips' real fps instead of 25.0)

Usage:
    python split_clips.py --in_dir my_clips/ --out_dir data/segments --stride 2
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--stride", required=True, type=int,
                    help="Must equal frames.auto_stride(fps) of your clips")
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--crf", type=int, default=12, help="x264 quality (lower = better)")
    args = ap.parse_args()

    seg_len = args.stride * args.num_frames
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.in_dir.glob("*.mp4"))
    if not videos:
        sys.exit(f"No MP4s in {args.in_dir}")

    total = 0
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n_seg = n_frames // seg_len  # the incomplete tail is dropped
        print(f"{video.name}: {n_frames} frames -> {n_seg} segments of {seg_len} frames")
        for k in range(n_seg):
            out = args.out_dir / f"{video.stem}_s{k:03d}.mp4"
            if out.exists():
                continue
            start = k * seg_len
            end = start + seg_len - 1
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                "-vf", f"select='between(n,{start},{end})',setpts=N/FRAME_RATE/TB",
                "-frames:v", str(seg_len), "-an",
                "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
                str(out),
            ]
            subprocess.run(cmd, check=True)
            total += 1
    print(f"done: {total} new segments in {args.out_dir}")


if __name__ == "__main__":
    main()