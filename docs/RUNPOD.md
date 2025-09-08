# Runpod Setup and Launch

Recommended base image: Runpod Pytorch 2.4.0
- Image tag: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Why:
  - Python 3.11 (широкая поддержка бинарных колёс: sentencepiece/librosa/etc.)
  - CUDA 12.4.1 (стабильные колёса PyTorch/торч-аудио)
  - Более консервативно и совместимо с NeMo, чем 2.8.0; свежее и удобнее, чем 2.2.0

## Bootstrap
1) В контейнере перейти к репо и выполнить:
```
bash transcribe/training/runpod_setup.sh
```
- Установит ffmpeg, обновит pip, поставит nemo_toolkit/lhotse/lightning и т.п.
- Настроит локальные кэши (`HF_HOME`, `TORCH_HOME`, `TMP`, `PIP_CACHE_DIR`).

2) Активировать окружение переменных (в каждой новой сессии):
```
source transcribe/env.sh
```

## Native NeMo LoRA (Canary)
Запуск дообучения с нативным адаптером NeMo (LoRA) на A6000 48 GB:
```
python transcribe/training/runpod_nemo_canary_lora.py \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train.jsonl \
  --val /workspace/data/val.jsonl \
  --outdir /workspace/exp/canary_ru_lora_a6000 \
  --export /workspace/models/canary-ru-lora-a6000.nemo \
  --bs 8 --accum 1 --precision bf16 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
```

### Автозагрузка Canary .nemo из HF
Если `.nemo` нет в контейнере, можно поручить скрипту скачать его из Hugging Face:
```
python transcribe/training/runpod_nemo_canary_lora.py \
  --auto_download --model_id nvidia/canary-1b-v2 --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train.jsonl --val /workspace/data/val.jsonl \
  --outdir /workspace/exp/canary_ru_lora_a6000 --export /workspace/models/canary-ru-lora-a6000.nemo \
  --bs 8 --accum 1 --precision bf16
```
Примечания:
- При наличии `HF_TOKEN` в окружении приватные модели тоже скачаются (если требуется).
- Файл будет сохранён в кэш HF, а также скопирован в указанный `--nemo`, если возможно.

Примечания
- Скрипт печатает телеметрию VRAM каждые 50 шагов. Если есть запас, можно поднять `--bs` до 12.
- Если ваш билд NeMo не поддерживает `add_adapter`, обновите `nemo_toolkit` или используйте временно Windows-скрипт с кастомной LoRA.
- Кэши/временные файлы хранятся в репо (`transcribe/.hf`, `.torch`, `.tmp`).
