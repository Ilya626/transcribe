# Speech Model Inference and Evaluation

This repository contains scripts to run speech recognition inference with several open-source models and to evaluate their quality.

## Inference

Each model has its own script under `models/`:

- `inference_canary.py` – [NVIDIA Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2)
- `inference_whisper.py` – [OpenAI Whisper large v3](https://huggingface.co/openai/whisper-large-v3)
- `inference_gigaam.py` – [Salute GigaAM v2](https://github.com/salute-developers/GigaAM)
- `inference_silero.py` – [Silero STT](https://github.com/snakers4/silero-models)
- `inference_vosk.py` – [Vosk](https://alphacephei.com/vosk)

Usage is similar for all scripts:

```bash
python models/inference_whisper.py <path/to/audio_or_dir> predictions.json
```

Results are written as JSON mapping audio paths to transcribed text.

For a broader overview of Russian speech recognition projects, see resources like the [Alpha Cephei blog](https://alphacephei.com/nsh/2025/04/18/russian-models.html) which surveys Vosk, Silero, Whisper and other models.

## Evaluation

To compare predictions with reference transcripts, use:

```bash
python evaluation/evaluate_transcriptions.py reference.json predictions.json
```

The script prints word error rate (WER), character error rate (CER) and a semantic similarity score using Sentence-Transformers.

To evaluate several model outputs in one run:

```bash
python evaluation/compare_transcriptions.py reference.json whisper.json canary.json gigaam.json
```

This prints metrics for each predictions file, allowing quick comparison between models.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
# For GigaAM
pip install git+https://github.com/salute-developers/GigaAM
```

Ensure `ffmpeg` is available for audio decoding.

