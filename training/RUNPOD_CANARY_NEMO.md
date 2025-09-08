# Runpod: Canary LoRA (Native NeMo) — Quick Start

Recommended image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

## One-time setup per pod

```bash
# 0) In Runpod web terminal
cd /workspace

# 1) Clone your repo (replace with your URL)
# Example (HTTPS):
#   git clone https://github.com/<you>/<repo>.git
# Example (SSH):
#   git clone git@github.com:<you>/<repo>.git
GIT_URL=<REPO_URL>
REPO_DIR=$(basename "$GIT_URL" .git)
[ -d "$REPO_DIR" ] || git clone --depth 1 "$GIT_URL"
cd "$REPO_DIR"

# 2) (Optional) HF token if needed for gated/private models
# export HF_TOKEN=hf_xxx

# 3) Install deps (NeMo from Git main) and system packages
bash transcribe/training/runpod_setup_nemo_main.sh

# 4) Activate project-local caches for this shell
source transcribe/env.sh

# 5) Quick sanity
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available());\
          print('gpu',torch.cuda.get_device_name(0))"
```

## Start training (auto-download .nemo)

```bash
python transcribe/training/runpod_nemo_canary_lora.py \
  --auto_download --model_id nvidia/canary-1b-v2 \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train.jsonl \
  --val   /workspace/data/val.jsonl \
  --outdir /workspace/exp/canary_ru_lora_a6000 \
  --export /workspace/models/canary-ru-lora-a6000.nemo \
  --bs 8 --accum 1 --precision bf16 --num_workers 8 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
```

Notes
- If VRAM allows on A6000 (48 GB), try `--bs 12`.
- Resume training from last checkpoint:

```bash
python transcribe/training/runpod_nemo_canary_lora.py \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train.jsonl \
  --val   /workspace/data/val.jsonl \
  --outdir /workspace/exp/canary_ru_lora_a6000 \
  --export /workspace/models/canary-ru-lora-a6000.nemo \
  --bs 8 --accum 1 --precision bf16 --num_workers 8 \
  --resume /workspace/exp/canary_ru_lora_a6000/last.ckpt
```

Where
- Outputs/checkpoints: `/workspace/exp/canary_ru_lora_a6000/`
- Exported merged `.nemo`: `/workspace/models/canary-ru-lora-a6000.nemo`
- Caches: `transcribe/.hf`, `transcribe/.torch`, temps: `transcribe/.tmp`
