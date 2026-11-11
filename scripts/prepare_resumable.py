#!/usr/bin/env python3
"""Resumable, disk-friendly driver around scripts/prepare_dataset.py.

Uses the unchanged process_one() from prepare_dataset.py, but:
  * deletes the per-segment work folder (frames/pose/depth/masks/overlay) right
    after the segment is done - everything the LoRA training needs was already
    written to <out>/<variant>/{videos,vace_videos,reference_images_*}
  * appends each finished segment to metadata_vace.csv immediately, so a crash
    or Ctrl-C loses at most the segment in progress
  * on restart, skips segments that are already in the CSV (or were skipped
    on purpose, e.g. failed the mask-coverage check)
  * a crash in one segment does not stop the run; many failures in a row do

Place this file in the repo root (next to scripts/ and regenhuman/).
"""

import argparse
import csv
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import prepare_dataset  # noqa: E402  (scripts/prepare_dataset.py)
from regenhuman import config, segment  # noqa: E402

FIELDS = ["video", "prompt", "vace_video", "vace_reference_image", "vace_reference_image_gt"]


def read_lines(path):
    if not path.exists():
        return set()
    return {l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--captions", required=True, help="CSV with columns video_id,prompt")
    ap.add_argument("--variant", required=True, choices=config.VARIANTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segmenter", default="sam3", choices=segment.SEGMENTERS)
    ap.add_argument("--num_frames", type=int, default=config.NUM_FRAMES)
    ap.add_argument("--every", type=int, default=1,
                    help="Use only every Nth segment (thins near-duplicate neighbours)")
    ap.add_argument("--max_consecutive_failures", type=int, default=5)
    ap.add_argument("--keep_work", action="store_true", help="Do not delete work folders")
    args = ap.parse_args()

    captions = {}
    with open(args.captions, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            captions[row["video_id"]] = row["prompt"]

    videos = sorted(Path(args.videos_dir).glob("*.mp4"))
    if not videos:
        sys.exit(f"No MP4s in {args.videos_dir}")
    missing = [v.stem for v in videos if v.stem not in captions]
    if missing:
        print(f"[warn] {len(missing)} videos have no caption and are skipped: {missing[:5]}")
    videos = [v for v in videos if v.stem in captions][::max(1, args.every)]

    out_root = Path(args.out) / args.variant
    work_root = Path(args.out) / "_work"
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "metadata_vace.csv"
    skipped_path = out_root / "_skipped.txt"
    failed_path = out_root / "_failed.txt"

    done = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            done = {Path(r["video"]).stem for r in csv.DictReader(f)}
    skipped = read_lines(skipped_path)

    todo = [v for v in videos if v.stem not in done and v.stem not in skipped]
    print(f"{len(videos)} segments selected | {len(done)} already done | "
          f"{len(skipped)} skipped earlier | {len(todo)} to process", flush=True)

    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        csv_f.flush()

    n_ok = n_skip = n_fail = consecutive_fail = 0
    t0 = time.monotonic()
    try:
        for i, video in enumerate(todo, 1):
            vid = video.stem
            elapsed = time.monotonic() - t0
            eta = f", ETA {elapsed / (i - 1) * (len(todo) - i + 1) / 3600:.1f} h" if i > 1 else ""
            print(f"[{i}/{len(todo)}] {video.name}{eta}", flush=True)
            try:
                row = prepare_dataset.process_one(
                    video, captions[vid], args.variant, args.segmenter,
                    out_root, work_root, args.num_frames)
                if row:
                    writer.writerow(row)
                    csv_f.flush()
                    n_ok += 1
                else:
                    with open(skipped_path, "a", encoding="utf-8") as sf:
                        sf.write(vid + "\n")
                    n_skip += 1
                consecutive_fail = 0
            except Exception:
                traceback.print_exc()
                with open(failed_path, "a", encoding="utf-8") as ff:
                    ff.write(vid + "\n")
                n_fail += 1
                consecutive_fail += 1
                if consecutive_fail >= args.max_consecutive_failures:
                    print(f"\nABORT: {consecutive_fail} failures in a row - likely an "
                          f"environment problem (CUDA libs? GPU memory? disk?). "
                          f"Fix it and re-run; finished segments are kept.")
                    break
            finally:
                if not args.keep_work:
                    shutil.rmtree(work_root / vid, ignore_errors=True)
    except KeyboardInterrupt:
        print("\ninterrupted - finished segments are kept, re-run to resume")
    finally:
        csv_f.close()
        if not args.keep_work:
            try:
                work_root.rmdir()  # only succeeds if empty
            except OSError:
                pass

    # integrity check: every CSV row must point to existing files
    bad = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for key in ("video", "vace_video", "vace_reference_image"):
                if not (out_root / r[key]).exists():
                    bad += 1
                    print(f"[warn] missing file for {r['video']}: {r[key]}")
    total_rows = len(done) + n_ok
    print(f"\nthis run: {n_ok} ok, {n_skip} skipped, {n_fail} failed | "
          f"CSV now has {total_rows} examples | broken file refs: {bad}")
    if n_fail:
        print(f"failed segments are listed in {failed_path} (retried on the next run)")
    print(f"Train with: DATASET_DIR={out_root} OUTPUT_DIR=outputs/{args.variant}_lora "
          f"bash scripts/train_lora.sh")


if __name__ == "__main__":
    main()