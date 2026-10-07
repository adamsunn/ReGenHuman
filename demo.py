#!/usr/bin/env python3
"""ReGenHuman demo: anonymize one video end-to-end.

Examples:
    python demo.py --input video.mp4 --variant structall --auto_caption
    python demo.py --input video.mp4 --variant structhuman \\
        --prompt "A man in a workshop assembles a wooden chair..." \\
        --segmenter grounded_sam2
"""

import argparse
import shutil
import sys
from pathlib import Path

from regenhuman import config, pipeline
from regenhuman.segment import SEGMENTERS


def preflight(args):
    from regenhuman.frames import _ffmpeg_bin

    problems = []
    ff = _ffmpeg_bin()
    if ff == "ffmpeg" and shutil.which("ffmpeg") is None:
        problems.append("no ffmpeg available (pip install imageio-ffmpeg, or put ffmpeg on PATH)")
    for p in (config.DWPOSE_DET_ONNX, config.DWPOSE_POSE_ONNX):
        if not p.exists():
            problems.append(f"DWPose checkpoint missing: {p} — run scripts/setup_models.sh")
            break
    if not config.VDA_CHECKPOINTS[args.depth_encoder].exists():
        problems.append(
            f"Video-Depth-Anything ({args.depth_encoder}) checkpoint missing — run scripts/setup_models.sh"
        )
    lora = Path(args.lora) if args.lora else config.default_lora_path(args.variant)
    if not lora.exists():
        problems.append(
            f"LoRA weights missing: {lora} — download weights.zip from the "
            "Google Drive link in the README and unzip it as weights/"
        )
    if problems:
        print("Preflight failed:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        sys.exit(1)
    if args.segmenter == "sam3":
        print(
            "Note: SAM3 weights are gated on Hugging Face (facebook/sam3). If the "
            "download fails, request access + `huggingface-cli login`, or use "
            "--segmenter grounded_sam2."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input MP4 video")
    ap.add_argument("--variant", required=True, choices=config.VARIANTS,
                    help="structall = regenerate everything; structhuman = keep background")
    ap.add_argument("--prompt", default=None,
                    help="Dense caption of the scene/action (recommended ~120-160 words)")
    ap.add_argument("--auto_caption", action="store_true",
                    help="Generate the prompt with Qwen3-VL-8B-Instruct")
    ap.add_argument("--segmenter", default="sam3", choices=SEGMENTERS)
    ap.add_argument("--lora", default=None, help="LoRA .safetensors (default: shipped weights)")
    ap.add_argument("--output", default=None, help="Output MP4 (default: <work_dir>/output.mp4)")
    ap.add_argument("--work_dir", default=None,
                    help="Intermediate artifacts dir (default: runs/<input-stem>_<variant>/)")
    ap.add_argument("--num_frames", type=int, default=config.NUM_FRAMES,
                    help=f"Max frames to sample (default {config.NUM_FRAMES}; longer inputs "
                         "are generated in overlapping chunks if you raise this)")
    ap.add_argument("--frame_stride", type=int, default=None,
                    help="Sample every Nth frame (default: auto to ~7.5 fps)")
    ap.add_argument("--width", type=int, default=config.WIDTH)
    ap.add_argument("--height", type=int, default=config.HEIGHT)
    ap.add_argument("--steps", type=int, default=config.NUM_INFERENCE_STEPS)
    ap.add_argument("--cfg", type=float, default=config.CFG_SCALE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--fps_out", type=float, default=config.OUTPUT_FPS)
    ap.add_argument("--chunk_size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--frame_overlap", type=int, default=config.FRAME_OVERLAP)
    ap.add_argument("--depth_encoder", default="vitl", choices=("vitl", "vits"),
                    help="vitl matches training (weights CC-BY-NC); vits is Apache-licensed")
    ap.add_argument("--wan_dir", default=None,
                    help="Pre-downloaded Wan2.1-VACE-1.3B dir (skips auto-download)")
    ap.add_argument("--resume", action="store_true", help="Reuse existing stage outputs")
    args = ap.parse_args()

    if not args.prompt and not args.auto_caption:
        ap.error("provide --prompt or --auto_caption (generation quality degrades "
                 "badly without a dense caption)")

    input_video = Path(args.input)
    if not input_video.exists():
        ap.error(f"input video not found: {input_video}")
    work_dir = Path(args.work_dir or Path("runs") / f"{input_video.stem}_{args.variant}")

    preflight(args)

    pipeline.run(
        input_video=input_video,
        variant=args.variant,
        work_dir=work_dir,
        prompt=args.prompt,
        auto_caption=args.auto_caption,
        segmenter=args.segmenter,
        lora_path=args.lora,
        output_path=args.output,
        wan_dir=args.wan_dir,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        fps_out=args.fps_out,
        chunk_size=args.chunk_size,
        frame_overlap=args.frame_overlap,
        depth_encoder=args.depth_encoder,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
