# Fine-tuning

## Whisper (experimental)
- Script: `transcribe/training/finetune_whisper.py`
- Expects JSON manifests (lists) with fields `audio` (wav path) and `text`:
```
python transcribe/training/finetune_whisper.py train.json eval.json out_dir \
  --model_id openai/whisper-small --batch_size 2 --num_epochs 1
```
- Adjust model size and batch params for your GPU.

## Canary (LoRA via NeMo)
- Script: `transcribe/training/finetune_canary.py`
- Expects JSONL with `audio_filepath` and `text`:
```
python transcribe/training/finetune_canary.py --nemo path/to/model.nemo \
  --train data/train.jsonl --val data/val.jsonl
```
- Injects LoRA, trains, and exports a merged `.nemo` checkpoint.

## Canary — LoRA (Runpod/Windows)

- Script: `transcribe/training/runpod_nemo_canary_lora.py`
- Purpose: lightweight adapter training (LoRA). Falls back to custom Linear-wrapping LoRA if native NeMo adapters are unavailable.
- Key flags:
  - Presets: `--preset a6000-fast|a6000-max` (CLI flags override)
  - Logging: `--log csv|tb`; TensorBoard appears under `<outdir>/tb_logs`
  - Early stop / best model: `--early_stop --es_patience N --es_min_delta d`, best checkpoint saved as `best.ckpt`
  - Optimizer overrides: `--lr`, `--weight_decay`, `--fused_optim`
  - Scheduler: `--sched cosine|inverse_sqrt|none`, with `--warmup_steps|--warmup_ratio`, `--min_lr`
  - Safety: `--gradient_clip_val` (default 1.0)

Example (A6000 48 GB):
```
python transcribe/training/runpod_nemo_canary_lora.py \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train_portable.jsonl \
  --val   /workspace/data/val_portable.jsonl \
  --outdir /workspace/exp/canary_ru_lora \
  --export /workspace/models/canary-ru-lora.nemo \
  --preset a6000-max --log tb --early_stop --es_patience 4 --es_min_delta 0.003
```

## Canary — Partial Unfreezing (deeper FT)

- Script: `transcribe/training/runpod_nemo_canary_partial.py`
- Purpose: более глубокое дообучение за счёт частичной разморозки слоёв энкодера/декодера и головы; по желанию можно добавить гибридную LoRA.
- Defaults: все веса заморожены; далее размораживаются выбранные подсекции.
- Key flags:
  - Разморозка: `--unfreeze_encoder_last N`, `--unfreeze_decoder_last M`, `--unfreeze_head`, `--train_norms`, `--train_bias`, `--grad_ckpt`
  - Гибридная LoRA: `--with_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05`
  - Пресеты/обучение/валидация: те же, что в LoRA‑скрипте (`--preset`, `--log`, `--early_stop`, `--es_*`, `--lr`, `--weight_decay`, `--gradient_clip_val`)

Examples:
```
# E4 + D2 + head (bf16), ранняя остановка
python transcribe/training/runpod_nemo_canary_partial.py \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train_portable.jsonl \
  --val   /workspace/data/val_portable.jsonl \
  --outdir /workspace/exp/canary_partial_e4_d2 \
  --export /workspace/models/canary-partial-e4-d2.nemo \
  --unfreeze_encoder_last 4 --unfreeze_decoder_last 2 --unfreeze_head \
  --preset a6000-fast --early_stop --es_patience 4 --es_min_delta 0.003

# Гибрид (partial + LoRA)
python transcribe/training/runpod_nemo_canary_partial.py \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train_portable.jsonl \
  --val   /workspace/data/val_portable.jsonl \
  --outdir /workspace/exp/canary_partial_hybrid \
  --export /workspace/models/canary-partial-hybrid.nemo \
  --unfreeze_encoder_last 4 --unfreeze_decoder_last 2 --unfreeze_head \
  --with_lora --lora_r 16 --lora_alpha 32 --preset a6000-fast
```

## Best‑practice (суммарно)

- Валидация и ранняя остановка: логируйте `val_loss` и сохраняйте `best.ckpt` по минимуму; `--early_stop` c patience 3–5 и min_delta ≈0.3–0.5%.
- Дожим от лучшего чекпойнта: снижайте `--lr` в 2–3 раза (например, 3e‑5 → 1e‑5), пробегите ещё 10–20% шагов.
- План обучения: inverse_sqrt (Noam‑подобный) или cosine с `min_lr≈0.1×base` и `warmup_steps/ratio` 3–5%.
- Стабильность: `--gradient_clip_val 1.0`, при спайках увеличьте `--accum`, проверьте аугментации и сверхдлинные семплы.
- Регуляризация: `--weight_decay 0.01`, у LoRA можно поднять `--lora_dropout` до 0.1 (проверяйте валид).
- Данные: используйте переносные JSONL и контролируйте max_duration самплера (`--max_duration`), включайте `pin_memory` и `persistent_workers`.
