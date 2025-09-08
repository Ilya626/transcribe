# Inference Runbook

## 0. Activate env (PowerShell)
- `. transcribe/env.ps1`
- `.\\transcribe\\.venv\\Scripts\\Activate.ps1`

## 1. Whisper
- Large v3: `python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/whisper_large_v3_train_beam2_bs8.json --model_id openai/whisper-large-v3 --language ru --task transcribe --num_beams 2 --batch_size 8`
- Large v3 turbo: `python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/whisper_large_v3_turbo_train_beam2_bs10.json --model_id openai/whisper-large-v3-turbo --language ru --task transcribe --num_beams 2 --batch_size 10`
- Distil large v3.5: `python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/distil_large_v35_train.json --model_id distil-whisper/distil-large-v3.5 --language ru --task transcribe --num_beams 1 --batch_size 64`

## 2. Canary (NVIDIA)
- `python transcribe/models/inference_canary.py data/train.jsonl transcribe/preds/canary_train_full_asr.json --model_id nvidia/canary-1b-v2 --language ru --task transcribe --batch_size 24`

## 3. GigaAM (Salute)
- Install once: `pip install git+https://github.com/salute-developers/GigaAM`
- Example full-train with parallel shards (3):
  - `python transcribe/models/inference_gigaam.py data/train.jsonl transcribe/preds/gigaam_train_full_par3.json --model v2_rnnt --parallel 3`

## 4. Optional: Silero / Vosk
- Silero: `python transcribe/models/inference_silero.py <audio_or_dir> transcribe/preds/silero.json`
- Vosk: `python transcribe/models/inference_vosk.py <audio_or_dir> transcribe/preds/vosk.json`

## Tips
- Add `--engine pipeline` for Whisper to chunk long-form (`chunk_length_s=30`).
- Always set `--language ru --task transcribe` for Whisper/Canary to avoid auto-translate.
- Scripts enforce a simple GPU lock to avoid concurrent runs.
