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
