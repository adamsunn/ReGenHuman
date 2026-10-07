# third_party/

Populated by setup scripts; nothing here is committed.

- `dwpose_ckpts/` — DWPose ONNX checkpoints (`scripts/setup_models.sh`)
- `Video-Depth-Anything/` — cloned @ `859cce4` with its `checkpoints/` (`scripts/setup_models.sh`)
- `sam2_ckpts/` — SAM2.1 hiera-tiny (auto-downloaded on first `--segmenter grounded_sam2` use)
- `DiffSynth-Studio/` — patched clone for TRAINING only (`scripts/apply_diffsynth_patch.sh`)

The vendored DWPose *inference code* lives in `regenhuman/dwpose_onnx/`
(Apache-2.0, see the LICENSE file there).
