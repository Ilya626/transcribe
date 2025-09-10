# Models Reference

## Whisper (transformers)
- Script: `transcribe/models/inference_whisper.py`
- Key args:
  - `--model_id`: e.g., `openai/whisper-large-v3`, `openai/whisper-large-v3-turbo`, `distil-whisper/distil-large-v3.5`
  - `--language ru --task transcribe` (recommended)
  - `--batch_size`, `--num_beams`, `--engine {generate|pipeline}`
- Defaults (12GB GPU): large-v3 bs=24; turbo bs=64; distil bs=64.

## Canary (NVIDIA)
- Script: `transcribe/models/inference_canary_nemo.py`
- `--model_id nvidia/canary-1b-v2` downloads the `.nemo` from HF (requires `HF_TOKEN`)
- Use `--nemo path/to/model.nemo` to run a local checkpoint
- `--source_lang ru --target_lang ru --task asr` for transcription; adjust `--batch_size` as needed

## GigaAM (Salute)
- Script: `transcribe/models/inference_gigaam.py`
- Install: `pip install git+https://github.com/salute-developers/GigaAM`
- Model selector: `--model v2_rnnt|v2_ctc|...`
- Can run parallel shards: `--parallel N` (merges outputs)
- [Detailed guide](GIGAAM.md)

## Silero / Vosk
- Silero: `transcribe/models/inference_silero.py`
- Vosk: `transcribe/models/inference_vosk.py`

## Outputs
- All inference scripts write a JSON mapping `audio_path -> text`.
- For JSONL input, the keys match `audio_filepath` values.
