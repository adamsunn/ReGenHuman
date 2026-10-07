# Training your own ReGenHuman LoRA

The released weights were trained on HOIGen-1M, which we do not
redistribute — bring your own videos and captions. The full recipe below is
exactly what produced `weights/*/step-16000.safetensors`.

## 1. Data requirements

- **Videos**: MP4s containing people, long enough to yield 49 frames at
  ~7.5 fps (≈6.5 s at 30 fps source). Clips that fail this, or in which the
  segmenter finds no person, are skipped automatically.
- **Captions**: one dense paragraph per video (~120–160 words) describing
  scene, people (clothing/appearance/position), actions over time, and
  camera framing. Short captions measurably hurt the model. Example of the
  kind of caption the released weights were trained on:

  > In the video, a man is seen working in a workshop environment. He is
  > wearing a black t-shirt and a face mask, indicating a focus on safety
  > and hygiene. The man is operating a machine that appears to be a
  > metalworking device, possibly a press or a cutting machine, as he is
  > handling metal sheets. The workshop is well-equipped with various tools
  > and machinery, including a computer monitor in the background [...] The
  > man's actions are methodical and focused, demonstrating expertise in
  > the task at hand.

  A CSV with columns `video_id,prompt` (where `video_id` is the file name
  without `.mp4`).

## 2. Build the dataset

```bash
python scripts/prepare_dataset.py \
    --videos_dir my_videos/ --captions captions.csv \
    --variant structall --out data/my_dataset [--segmenter sam3]
```

This runs the same conditioning extraction as the demo (frames → DWPose →
depth → human masks → overlay) and writes:

| CSV column | Content |
|---|---|
| `video` | ground-truth 49-frame clip (`videos/{id}.mp4`) |
| `prompt` | the caption |
| `vace_video` | conditioning clip (`vace_videos/{id}_vace.mp4`) |
| `vace_reference_image` | first conditioning frame — **used in training** |
| `vace_reference_image_gt` | first real frame (kept for the reconstruction ablation; unused by default) |

Repeat with `--variant structhuman` for the second adapter: the two variants
are **separate LoRAs** trained on their own conditioning.

## 3. Patched DiffSynth-Studio

```bash
bash scripts/apply_diffsynth_patch.sh   # clone @ a6884f6 + apply patches/
```

Training (not inference) needs our 5-file patch:
- `LoadVideo` caps decoded frames at `--num_frames` and retries flaky
  network-filesystem reads (imageio's 10 s ffmpeg timeout);
- a VACE token-count sanity check (fails fast on resolution mismatches);
- `--resume_checkpoint` (warm-start from a saved `step-*.safetensors`);
- `--ref_drop_prob` (reference-image dropout, off by default);
- `--latent_cache_dir` (optional pre-encoded VAE latents).

## 4. Train

```bash
DATASET_DIR=data/my_dataset/structall OUTPUT_DIR=outputs/structall_lora GPUS=2 \
    bash scripts/train_lora.sh
```

Recipe (identical to the released weights): rank-32 LoRA, alpha 32, targets
`q,k,v,o,ffn.0,ffn.2` on the VACE context adapter only (backbone, VAE and
text encoder frozen), lr 1e-4, 20 epochs, batch 1 per GPU, 480x832,
49 frames, checkpoint every 2,000 steps. `--num_frames 49` is passed
explicitly — DiffSynth's default is 81 and will silently mismatch your
49-frame clips. One 46 GB GPU is enough
(`--use_gradient_checkpointing_offload` is on); at ~15 s/step, 16k steps is
roughly 3 GPU-days — scale `GPUS` to taste.

## 5. Evaluate a checkpoint

Checkpoints land as `OUTPUT_DIR/step-{N}.safetensors` with the `pipe.vace.`
prefix already stripped, so they load directly:

```bash
python demo.py --input clip.mp4 --variant structall \
    --lora outputs/structall_lora/step-16000.safetensors --auto_caption
```