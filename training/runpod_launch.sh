#!/usr/bin/env bash
set -euo pipefail

# One-shot launcher for Runpod: setup env, start TensorBoard (optional) and run training.

usage() {
  cat <<'USAGE'
Usage:
  bash transcribe/training/runpod_launch.sh \
    --train /workspace/data/train_portable.jsonl \
    --val   /workspace/data/val_portable.jsonl \
    [--preset a6000-max|a6000-fast] [--log tb|csv] [--tb_port 8888] [--stop_jupyter] \
    [--nemo /workspace/models/canary-1b-v2.nemo] [--auto_download] [--model_id nvidia/canary-1b-v2]

Notes:
  - Default preset is a6000-max. Set --preset a6000-fast for safer settings.
  - If --log tb and --tb_port specified, TensorBoard will be launched on 0.0.0.0:PORT.
  - --stop_jupyter kills Jupyter to free port 8888 when reusing it for TensorBoard.
USAGE
}

PRESET="a6000-max"
LOGFMT="csv"
TB_PORT="8888"
STOP_JUPYTER=false
TRAIN=""
VAL=""
NEMO="/workspace/models/canary-1b-v2.nemo"
MODEL_ID="nvidia/canary-1b-v2"
AUTO_DL=false
OUTDIR="/workspace/exp/canary_ru_lora_auto"
EXPORT="/workspace/models/canary-ru-lora-auto.nemo"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) TRAIN="$2"; shift 2;;
    --val) VAL="$2"; shift 2;;
    --preset) PRESET="$2"; shift 2;;
    --log) LOGFMT="$2"; shift 2;;
    --tb_port) TB_PORT="$2"; shift 2;;
    --stop_jupyter) STOP_JUPYTER=true; shift;;
    --nemo) NEMO="$2"; shift 2;;
    --model_id) MODEL_ID="$2"; shift 2;;
    --auto_download) AUTO_DL=true; shift;;
    --outdir) OUTDIR="$2"; shift 2;;
    --export) EXPORT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$TRAIN" || -z "$VAL" ]]; then
  echo "[ERR] --train and --val are required" >&2
  usage
  exit 1
fi

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/env.sh" >/dev/null 2>&1 || true

if $STOP_JUPYTER; then
  (pkill -f jupyter || true) 2>/dev/null || true
  if command -v lsof >/dev/null 2>&1; then
    kill -9 $(lsof -t -i:${TB_PORT}) 2>/dev/null || true
  fi
fi

if [[ "$LOGFMT" == "tb" ]]; then
  python -m pip install -q -U tensorboard >/dev/null 2>&1 || true
fi

AUTO_FLAG=()
if $AUTO_DL; then AUTO_FLAG=(--auto_download --model_id "$MODEL_ID"); fi

echo "[LAUNCH] Training preset=$PRESET log=$LOGFMT outdir=$OUTDIR export=$EXPORT"
python "$HERE/training/runpod_nemo_canary_lora.py" \
  "${AUTO_FLAG[@]}" \
  --nemo "$NEMO" \
  --train "$TRAIN" \
  --val "$VAL" \
  --outdir "$OUTDIR" \
  --export "$EXPORT" \
  --preset "$PRESET" \
  --log "$LOGFMT"

if [[ "$LOGFMT" == "tb" ]]; then
  LOGDIR="$OUTDIR/tb_logs"
  echo "[TB] Launching TensorBoard on 0.0.0.0:${TB_PORT}, logdir=$LOGDIR"
  nohup tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port "$TB_PORT" >/workspace/tensorboard.out 2>&1 &
  echo "[TB] Open via Runpod UI (HTTP Services) port ${TB_PORT} or SSH port-forward"
fi

echo "[DONE]"

