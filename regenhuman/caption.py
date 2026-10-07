"""Optional dense captioning with Qwen3-VL-8B-Instruct.

The LoRAs were trained with dense ~120-160-word captions; generation quality
degrades with short or generic prompts. When the user has no caption, this
produces one from the original video frames. The model (~17 GB in bf16) is
loaded, run once, and fully unloaded before the diffusion stage.
"""

import gc
from pathlib import Path

from . import config

_INSTRUCTION = (
    "Describe this video in a single dense paragraph of 120-160 words. "
    "Cover: the scene and setting, every visible person (their clothing, "
    "appearance, and position), the actions they perform over time and the "
    "objects they interact with, and the camera framing. Write in flowing "
    "prose, present tense, no lists and no preamble."
)
_NUM_FRAMES = 8


def caption_video(frames_dir: Path) -> str:
    """Caption a video from its extracted frames; frees the VLM before returning."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No JPG frames in {frames_dir}")
    idx = [round(i * (len(frame_files) - 1) / (_NUM_FRAMES - 1)) for i in range(_NUM_FRAMES)]
    images = [Image.open(frame_files[i]).convert("RGB") for i in sorted(set(idx))]

    processor = AutoProcessor.from_pretrained(config.CAPTION_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        config.CAPTION_MODEL, dtype=torch.bfloat16, device_map="cuda",
    )

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": _INSTRUCTION})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    text = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )[0].strip()

    del model, processor, inputs, out
    gc.collect()
    torch.cuda.empty_cache()
    return text
