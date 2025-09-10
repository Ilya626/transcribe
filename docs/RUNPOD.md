# Runpod Setup and Launch

Recommended base image
- `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Rationale: Python 3.11 + CUDA 12.4 have stable wheels (NeMo/Lhotse/TorchAudio).

## Bootstrap
- Option A (PyPI NeMo):
```
bash transcribe/training/runpod_setup.sh
source transcribe/env.sh
export HF_TOKEN=hf_xxx    # required for gated HF datasets/models
# ensure your token has accepted https://huggingface.co/nvidia/canary-1b-v2
```
- Option B (NeMo from Git main):
```
bash transcribe/training/runpod_setup_nemo_main.sh
source transcribe/env.sh
export HF_TOKEN=hf_xxx    # token must have access to https://huggingface.co/nvidia/canary-1b-v2
```
Use Option B if you want latest NeMo features.

## Dataset (portable)
- Pack locally with `transcribe/tools/pack_dataset.py`.
- Upload to `/workspace/data` so you have:
  - `/workspace/data/train_portable.jsonl`
  - `/workspace/data/val_portable.jsonl`
  - `/workspace/data/Train data/` (audio files)

Ways to upload
- runpodctl (Windows PowerShell):
  - Compress-Archive -LiteralPath 'data\train_portable.jsonl','data\val_portable.jsonl','data\Train data' -DestinationPath dataset_portable.zip -Force
  - .\runpodctl.exe cp .\dataset_portable.zip "$POD:/workspace/"
  - .\runpodctl.exe exec $POD -- bash -lc "cd /workspace && apt-get update -y && apt-get install -y unzip >/dev/null 2>&1 || true; unzip -q -o dataset_portable.zip -d /workspace"
- SSH SCP (from your host):
  - scp -P <port> -i <key> dataset_portable.zip root@<host>:/workspace/
  - ssh -p <port> -i <key> root@<host> "cd /workspace && apt-get update -y && apt-get install -y unzip >/dev/null 2>&1 || true; unzip -q -o dataset_portable.zip -d /workspace"
- Hugging Face private repo (using `fetch_dataset.py`):
```
export HF_TOKEN=hf_xxx
python transcribe/tools/fetch_dataset.py --hf_repo <org>/<repo> --hf_file dataset_portable.tgz --dest /workspace/data
```

## Native NeMo LoRA (training)
Direct command (example, aggressive preset):
```
python transcribe/training/runpod_nemo_canary_lora.py \
  --auto_download --model_id nvidia/canary-1b-v2 \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train_portable.jsonl \
  --val   /workspace/data/val_portable.jsonl \
  --outdir /workspace/exp/canary_ru_lora_a6000 \
  --export /workspace/models/canary-ru-lora-a6000.nemo \
  --preset a6000-max
```
Notes:
- Presets: `a6000-fast` (bs=12) and `a6000-max` (bs=16). CLI flags override preset values.
- Fallback LoRA injection engages automatically if native adapters are unavailable.
- Safe allocator is enabled by default.

Validation, best checkpoint, early stopping
- Add `--early_stop --es_patience 4 --es_min_delta 0.003` (monitor defaults to `val_loss`).
- Best checkpoint `best.ckpt` is saved in `--outdir`.
- Resume from best if LoRA shape unchanged: `--resume <outdir>/best.ckpt`.
- Optional overrides: `--lr 1e-5`, `--weight_decay 0.01`.

## One‑shot launcher (with monitoring)
Use the helper to run training and optionally launch TensorBoard:
```
bash transcribe/training/runpod_launch.sh \
  --train /workspace/data/train_portable.jsonl \
  --val   /workspace/data/val_portable.jsonl \
  --preset a6000-max --log tb --tb_port 8888 \
  --auto_download --model_id nvidia/canary-1b-v2 \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --outdir /workspace/exp/canary_ru_lora_auto \
  --export /workspace/models/canary-ru-lora-auto.nemo
```
- If port 8888 is busy (Jupyter), add `--stop_jupyter`.
- Open via Connect → HTTP Services → port 8888 → Open.
- For CSV logs (without TB), set `--log csv`.

## Monitoring & tips
- TensorBoard manual start:
```
pip install -U tensorboard
EXP=/workspace/exp/<your_exp>
tensorboard --logdir "$EXP/tb_logs" --host 0.0.0.0 --port 6006 &
```
- CSV tail in console:
```
tail -n 20 -f /workspace/exp/<your_exp>/pl_logs/version_*/metrics.csv
```
- GPU/memory:
```
watch -n 3 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -i 0"
```

## Troubleshooting
- Permission denied running script: execute with `python ...` or fix LF + exec bit:
  - `sed -i 's/\r$//' transcribe/training/runpod_nemo_canary_lora.py && chmod +x transcribe/training/runpod_nemo_canary_lora.py`
- Hugging Face gated models: set `HF_TOKEN` before launching.
- Ports: reuse 8888 (stop Jupyter) or add a new HTTP Service port (e.g. 6006).
- Paths: training/inference resolve relative paths in JSONL against the JSONL file location.

