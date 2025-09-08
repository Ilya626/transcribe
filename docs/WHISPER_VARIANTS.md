# Whisper Variants: Adding & Running

This guide explains how to add a new Whisper-compatible model and run it on your data.

## Prerequisites
- Activate env: `. transcribe/env.ps1; .\\transcribe\\.venv\\Scripts\\Activate.ps1`
- GPU + CUDA available (CPU inference disabled)
- References in JSONL/JSON as described in `transcribe/docs/DATA.md`

## 1) Identify the model
- Find the Hugging Face `model_id` (e.g., `openai/whisper-large-v3`, a fine-tune like `user/ru-whisper-large-v3`, etc.).
- Compatibility: works with Whisper-family checkpoints that load via `AutoModelForSpeechSeq2Seq` + `AutoProcessor`.

## 2) First run on a subset
- Pick a small manifest to validate settings (batch size, beams). Example:
```
python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/<tag>_head.json \
  --model_id <MODEL_ID> --language ru --task transcribe --batch_size 8 --num_beams 1
```
- Tip: For long audio, add `--engine pipeline` to enable chunked decoding.

## 3) Full run (train.jsonl)
```
python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/<tag>_train.json \
  --model_id <MODEL_ID> --language ru --task transcribe --batch_size <B> --num_beams <N>
```
- Start with conservative settings if unknown: `--batch_size 16`, `--num_beams 1`.
- Heuristics for 12GB GPU:
  - large-v3: `--batch_size 24`
  - large-v3-turbo: `--batch_size 64`
  - distil-large-v3.5: `--batch_size 64`
- For RU fine-tunes of large-v3: begin with `--batch_size 8–16`; for turbo-based: `--batch_size 32–64` (reduce if OOM).

## 4) Naming convention
- Place outputs under `transcribe/preds/`.
- Use descriptive stems: `<family>_<variant>_train.json` (e.g., `whisper_large_v3_train.json`, `ru_whisper_large_v3_myft_train.json`).

## 5) Evaluate quality
- Quick metrics:
```
python transcribe/evaluation/compare_transcriptions.py data/train.jsonl \
  transcribe/preds/<tag>_train.json
```
- Advanced analysis (with baselines):
```
python -m transcribe.evaluation.analyze_errors data/train.jsonl \
  --pred base_v3=transcribe/preds/whisper_large_v3_train_beam2_bs8.json \
  --pred new=<path/to/your_output.json> \
  --outdir transcribe/preds/analysis_whisper_new --st_device cuda
```

## 6) Optional: add a default batch size rule
If you want the script to auto-pick a better `--batch_size` for your new model family, edit:
- `transcribe/models/inference_whisper.py` → function `_default_bs(mid: str)` and add a keyword match (e.g., part of `model_id`) returning a preferred integer.
Otherwise, always pass `--batch_size` explicitly.

## 7) Troubleshooting
- English words in RU output: ensure `--language ru --task transcribe` are set.
- OOM: lower `--batch_size`; avoid high `--num_beams`; consider `--engine pipeline`.
- Stuck GPU lock: remove `transcribe/.tmp/gpu.lock` only after ensuring no process uses the GPU.

