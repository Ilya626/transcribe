# Runpod: Canary LoRA (Native NeMo) — Quick Start

For base image selection, initial setup, and dataset upload, see [docs/RUNPOD.md](../docs/RUNPOD.md). Once the environment is prepared and `transcribe/env.sh` is sourced, launch training with:

## Start training (auto-download .nemo)

```bash
python transcribe/training/runpod_nemo_canary_lora.py \
  --auto_download --model_id nvidia/canary-1b-v2 \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train_portable.jsonl \
  --val   /workspace/data/val_portable.jsonl \
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
  --train /workspace/data/train_portable.jsonl \
  --val   /workspace/data/val_portable.jsonl \
  --outdir /workspace/exp/canary_ru_lora_a6000 \
  --export /workspace/models/canary-ru-lora-a6000.nemo \
  --bs 8 --accum 1 --precision bf16 --num_workers 8 \
  --resume /workspace/exp/canary_ru_lora_a6000/last.ckpt
```

Where
- Outputs/checkpoints: `/workspace/exp/canary_ru_lora_a6000/`
- Exported merged `.nemo`: `/workspace/models/canary-ru-lora-a6000.nemo`
- Caches: `transcribe/.hf`, `transcribe/.torch`, temps: `transcribe/.tmp`
