"""Wan2.1-VACE-1.3B generation with a ReGenHuman LoRA.

Loads the base model (auto-downloading into ``models/`` on first use — set
``DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`` to pull from Hugging Face instead
of ModelScope, or pass ``wan_dir`` for a pre-downloaded copy), applies the
42 MB LoRA to the VACE context adapter, and generates the anonymized video
from the conditioning frames.
"""

import gc
from pathlib import Path
from typing import List

from . import chunked, config

WAN_REPO = "Wan-AI/Wan2.1-VACE-1.3B"
TOKENIZER_REPO = "Wan-AI/Wan2.1-T2V-1.3B"


def load_pipeline(lora_path: Path, wan_dir: Path = None):
    """Build the WanVideoPipeline and load the ReGenHuman LoRA."""
    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    if wan_dir is not None:
        base = Path(wan_dir)
        model_configs = [
            ModelConfig(path=str(base / "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=str(base / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=str(base / "Wan2.1_VAE.pth")),
        ]
        tokenizer_config = ModelConfig(path=str(base / "google" / "umt5-xxl"))
    else:
        root = str(config.WAN_MODELS_DIR)
        model_configs = [
            ModelConfig(model_id=WAN_REPO, origin_file_pattern="diffusion_pytorch_model.safetensors", local_model_path=root),
            ModelConfig(model_id=WAN_REPO, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", local_model_path=root),
            ModelConfig(model_id=WAN_REPO, origin_file_pattern="Wan2.1_VAE.pth", local_model_path=root),
        ]
        tokenizer_config = ModelConfig(
            model_id=TOKENIZER_REPO, origin_file_pattern="google/umt5-xxl/", local_model_path=root,
        )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    lora_path = Path(lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
    pipe.load_lora(pipe.vace, str(lora_path), alpha=config.LORA_ALPHA)
    return pipe


def generate_video(
    overlay_frames: List,          # PIL RGB frames at (WIDTH, HEIGHT)
    ref_img,                       # PIL RGB reference image (first overlay frame)
    prompt: str,
    lora_path: Path,
    output_path: Path,
    wan_dir: Path = None,
    num_inference_steps: int = config.NUM_INFERENCE_STEPS,
    cfg_scale: float = config.CFG_SCALE,
    seed: int = config.SEED,
    chunk_size: int = config.CHUNK_SIZE,
    frame_overlap: int = config.FRAME_OVERLAP,
    fps: float = config.OUTPUT_FPS,
) -> Path:
    """Generate the anonymized video and save it to ``output_path``."""
    import torch
    from diffsynth.utils.data import save_video

    pipe = load_pipeline(lora_path, wan_dir=wan_dir)

    video = chunked.generate(
        pipe=pipe,
        control_frames=overlay_frames,
        ref_img=ref_img,
        prompt=prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        seed=seed,
        chunk_size=chunk_size,
        frame_overlap=frame_overlap,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
    )
    if len(video) > len(overlay_frames):
        video = video[: len(overlay_frames)]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_path), fps=fps, quality=5)

    del pipe, video
    gc.collect()
    torch.cuda.empty_cache()
    return output_path
