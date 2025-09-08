# Setup & Environment

- Python: use 3.11 or 3.12 on Windows for best wheel coverage.
- Create venv: `py -3.12 -m venv transcribe\\.venv`
- Activate in PowerShell:
  - `. transcribe/env.ps1`
  - `.\\transcribe\\.venv\\Scripts\\Activate.ps1`
- Install deps: `pip install -r transcribe/requirements.txt`
- FFmpeg: ensure `ffmpeg` is in PATH (audio decoding).

## Local Caches (Windows)
- `transcribe/env.ps1` sets:
  - `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_HUB_CACHE`
  - `TORCH_HOME`, `TMP`, `TEMP`, `PIP_CACHE_DIR`
- All go under `transcribe/.hf`, `transcribe/.torch`, `transcribe/.tmp`, `transcribe/.pip-cache`.
- Scope is current PowerShell session. Re-run `. transcribe/env.ps1` in new shells.

## GPU & CUDA
- GPU required for inference; CPU is disabled by design.
- Check CUDA: `python transcribe/tmp/check_cuda.py`
- Restrict GPU: set `CUDA_VISIBLE_DEVICES=0` before running scripts.
- Simple GPU lock prevents parallel runs: `transcribe/.tmp/gpu.lock`.
  - If a crash leaves a stale lock, remove it only after verifying no process is using the GPU.

## Known Warnings
- Transformers warns that `TRANSFORMERS_CACHE` is deprecated — harmless; `HF_HOME` is the primary root.
- HF Hub on Windows can warn about symlinks; enable Developer Mode to suppress.
