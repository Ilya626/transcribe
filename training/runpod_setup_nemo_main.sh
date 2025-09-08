#!/usr/bin/env bash
# Bootstrap for Runpod PyTorch 2.4.0 container using NeMo from Git (main).
# Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

set -euo pipefail

echo "[SYS] Installing system deps (ffmpeg, git, build-essential, cmake, libsndfile1)..."
apt-get update -y && apt-get install -y --no-install-recommends \
  ffmpeg git build-essential cmake libsndfile1 && rm -rf /var/lib/apt/lists/*

echo "[PIP] Upgrading pip/setuptools/wheel..."
python -m pip install -U pip setuptools wheel packaging ninja

echo "[PIP] Installing Python deps (from Git main for NeMo)..."
python - <<'PY'
import subprocess, sys
def pip(*args):
  subprocess.check_call([sys.executable, '-m', 'pip'] + list(args))

# Install NeMo from Git main with ASR extras
pip('install', '-U', 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]')

# Core libs
pip('install', '-U', 'lightning', 'lhotse', 'transformers', 'sentence-transformers',
    'soundfile', 'jiwer', 'einops', 'pandas', 'librosa', 'editdistance', 'webdataset', 'huggingface_hub')

print('[OK] NeMo (main) and deps installed')
PY

echo "[ENV] Configuring project-local caches..."
source "$(dirname "$0")/../env.sh"

echo "[CUDA] Torch: " $(python -c "import torch;print(torch.__version__, torch.cuda.is_available())")
python -c "import nemo_toolkit, nemo.collections.asr as asr; print('NeMo OK, ASR imported')"
echo "[READY] Setup (NeMo main) complete."

