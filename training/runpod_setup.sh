#!/usr/bin/env bash
# Bootstrap dependencies inside a Runpod PyTorch container (Ubuntu 22.04).
# Target image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

set -euo pipefail

echo "[SYS] Updating apt and installing ffmpeg..."
apt-get update -y && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

echo "[PIP] Upgrading pip/setuptools/wheel..."
python -m pip install -U pip setuptools wheel

echo "[PIP] Installing Python deps..."
python - <<'PY'
import subprocess, sys
pkgs = [
  'nemo_toolkit',
  'lightning',
  'lhotse',
  'soundfile', 'jiwer', 'einops', 'pandas', 'librosa', 'editdistance', 'webdataset',
  'transformers', 'sentence-transformers'
]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U'] + pkgs)
print('[OK] Python deps installed')
PY

echo "[ENV] Configuring project-local caches..."
source "$(dirname "$0")/../env.sh"

echo "[CUDA] Torch: " $(python -c "import torch;print(torch.__version__, torch.cuda.is_available())")
# Validate NeMo import (package name is 'nemo')
python - <<'PY'
try:
  import nemo
  from nemo.collections import asr
  print('[OK] NeMo imported:', getattr(nemo, '__version__', 'unknown'))
except Exception as e:
  print('[WARN] NeMo import check failed:', e)
PY
echo "[READY] Setup complete."
