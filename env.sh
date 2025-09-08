#!/usr/bin/env bash
# Project-local caches for Linux shells (Runpod, etc.)

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export HF_HOME="$HERE/.hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$HERE/.torch"
export TMPDIR="$HERE/.tmp"
export TMP="$HERE/.tmp"
export TEMP="$HERE/.tmp"
export PIP_CACHE_DIR="$HERE/.pip-cache"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$TMPDIR" "$PIP_CACHE_DIR"

echo "Project-local environment configured:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  TMP=$TMP"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"

