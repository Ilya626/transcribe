#!/usr/bin/env bash
set -euo pipefail

# Runpod/Linux launcher for Canary LoRA fine-tuning on A6000 (48 GB)
# Uses transcribe/training/finetune_canary.py and installs missing deps.

usage() {
  cat <<'USAGE'
Usage: runpod_finetune_canary.sh --nemo <path.nemo> --train <train.jsonl> --val <val.jsonl> \
  [--outdir <dir>] [--export <out.nemo>] [--bs 8] [--accum 1] \
  [--precision bf16] [--lora_r 16] [--lora_alpha 32] [--lora_dropout 0.05]

Example:
  ./transcribe/training/runpod_finetune_canary.sh \
    --nemo /workspace/models/canary-1b-v2.nemo \
    --train /workspace/data/train.jsonl \
    --val /workspace/data/val.jsonl \
    --outdir /workspace/exp/canary_ru_lora_a6000 \
    --export /workspace/models/canary-ru-lora-a6000.nemo
USAGE
}

NEMO=""; TRAIN=""; VAL=""; OUTDIR="/workspace/exp/canary_ru_lora_a6000"; EXPORT="/workspace/models/canary-ru-lora-a6000.nemo"
BS=8; ACCUM=1; PREC="bf16"; LORA_R=16; LORA_A=32; LORA_D=0.05

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nemo) NEMO="$2"; shift 2;;
    --train) TRAIN="$2"; shift 2;;
    --val) VAL="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --export) EXPORT="$2"; shift 2;;
    --bs) BS="$2"; shift 2;;
    --accum) ACCUM="$2"; shift 2;;
    --precision) PREC="$2"; shift 2;;
    --lora_r) LORA_R="$2"; shift 2;;
    --lora_alpha) LORA_A="$2"; shift 2;;
    --lora_dropout) LORA_D="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -z "$NEMO" || -z "$TRAIN" || -z "$VAL" ]] && { echo "Missing required args"; usage; exit 1; }

echo "[ENV] Python: $(python -V 2>&1)"
python - <<'PY'
import sys
try:
  import torch
  print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
  if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
except Exception as e:
  print('torch import error:', e); sys.exit(0)
PY

# Ensure core libs (nemo_toolkit, lhotse, lightning) exist. Torch usually preinstalled in Runpod image.
python - <<'PY'
import importlib, subprocess, sys
need = []
for m in ['nemo_toolkit', 'lhotse', 'lightning']:
  try:
    importlib.import_module(m)
  except Exception:
    need.append(m)
if need:
  print('[PIP] Installing:', need)
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U'] + need)
else:
  print('[PIP] All required packages present')
PY

# Kick off training with aggressive defaults for A6000 48GB
python transcribe/training/finetune_canary.py \
  --nemo "$NEMO" \
  --train "$TRAIN" \
  --val "$VAL" \
  --bs "$BS" \
  --accum "$ACCUM" \
  --precision "$PREC" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_A" \
  --lora_dropout "$LORA_D" \
  --exp_dir "$OUTDIR" \
  --export_nemo "$EXPORT" \
  --mem_report_steps 50 \
  --log csv

echo "[DONE] Training launched; outputs under $OUTDIR; export target: $EXPORT"
