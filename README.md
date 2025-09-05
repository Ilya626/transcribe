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

The script prints word error rate (WER), character error rate (CER), sentence error rate (SER) and a semantic similarity score. It
also reports counts of substitutions, insertions and deletions.

To evaluate several model outputs in one run:

```bash
python evaluation/compare_transcriptions.py reference.json whisper.json canary.json gigaam.json
```

This prints the same set of metrics for each predictions file, allowing quick comparison between models.

## Finetuning

Experimental fine-tuning of Whisper models is supported via
`training/finetune_whisper.py`.  It expects two JSON manifests with
fields `audio` (path to a WAV file) and `text` (reference transcript):

```bash
python training/finetune_whisper.py train.json eval.json out_dir \
    --model_id openai/whisper-small --batch_size 2 --num_epochs 1
```

The defaults aim at a Windows desktop with an RTX&nbsp;4070&nbsp;Ti (12&nbsp;GB).
Adjust the model size or batch parameters if you hit out-of-memory
errors.

For NVIDIA's Canary models, `training/finetune_canary.py` performs
LoRA-based fine-tuning using the NeMo toolkit and Lhotse manifests.  It
requires JSONL manifests with `audio_filepath` and `text` fields:

```bash
python training/finetune_canary.py --nemo path/to/model.nemo \
    --train train.jsonl --val val.jsonl
```

The script injects LoRA adapters, runs training and exports a merged
`.nemo` checkpoint on completion.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
# For GigaAM
pip install git+https://github.com/salute-developers/GigaAM
```

Ensure `ffmpeg` is available for audio decoding.

