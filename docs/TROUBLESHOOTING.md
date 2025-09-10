# Troubleshooting

## Common
- FFmpeg not found: install ffmpeg and ensure it's in PATH.
- CPU fallback blocked: scripts require CUDA; verify `torch.cuda.is_available()`.
- Stuck with GPU lock: delete `transcribe/.tmp/gpu.lock` only after confirming GPU is idle.

## Windows specifics
- Symlink warnings from HF Hub: enable Windows Developer Mode or ignore.
- Long paths / backslashes: prefer absolute paths in JSONL; use UTF-8 without BOM.
- `onnx` may fail to build on Python 3.13 — use Python 3.11/3.12 for prebuilt wheels.
- JSONL with BOM is handled automatically, but prefer plain UTF-8 without BOM.

## GigaAM
- Import error (sentencepiece): use Python 3.11/3.12; then `pip install gigaam` or the GitHub URL.
- Long-form audio issues: use `_chunk_transcribe` fallback (built into our script) or model's longform method.

## Whisper/Canary
- English tokens leak (the/and): ensure `--language ru --task transcribe` are set.
- OOM on large batches: reduce `--batch_size` and/or `--num_beams`; consider `--engine pipeline` for Whisper.

## Evaluation
- Slow semantic stage: set `--st_device cuda` and tweak `--st_bs`.
- Mojibake in references: fix upstream encoding and regenerate manifests to avoid training on corrupted text.
