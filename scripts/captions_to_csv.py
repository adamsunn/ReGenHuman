#!/usr/bin/env python3
"""Merge per-video caption .txt files into the captions.csv that
scripts/prepare_dataset.py expects (columns: video_id,prompt).

The captioning script names its outputs "{index:03d}_{video_stem}.txt", while
prepare_dataset.py needs video_id == the video file stem. This strips the
index prefix and checks that a matching video exists.

Usage:
    python captions_to_csv.py --captions_dir data/captions/txt \
        --videos_dir data/segments --out data/captions.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions_dir", required=True, type=Path)
    ap.add_argument("--videos_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows, orphans = [], []
    for txt in sorted(args.captions_dir.glob("*.txt")):
        video_id = re.sub(r"^\d+_", "", txt.stem, count=1)
        if not (args.videos_dir / f"{video_id}.mp4").exists():
            orphans.append(txt.name)
            continue
        prompt = " ".join(txt.read_text(encoding="utf-8").split())  # one line
        if prompt:
            rows.append({"video_id": video_id, "prompt": prompt})

    if not rows:
        sys.exit("No captions matched a video.")

    captioned = {r["video_id"] for r in rows}
    uncaptioned = [v.stem for v in args.videos_dir.glob("*.mp4") if v.stem not in captioned]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} captions -> {args.out}")
    if orphans:
        print(f"[warn] {len(orphans)} caption files without a matching video: {orphans[:5]}")
    if uncaptioned:
        print(f"[warn] {len(uncaptioned)} videos without caption (will be skipped): {uncaptioned[:5]}")


if __name__ == "__main__":
    main()