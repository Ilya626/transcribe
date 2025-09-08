# Evaluation & Analysis

## Quick metric check
- Compare a single predictions file: `python transcribe/evaluation/evaluate_transcriptions.py <ref.json|jsonl> <pred.json>`
- Compare multiple: `python transcribe/evaluation/compare_transcriptions.py data/train.jsonl transcribe/preds/whisper_large_v3_train_beam2_bs8.json transcribe/preds/canary_train_full_asr.json ...`

## Advanced analysis
- Command template:
```
python -m transcribe.evaluation.analyze_errors data/train.jsonl \
  --pred name1=path1.json \
  --pred name2=path2.json \
  --outdir transcribe/preds/analysis_<tag> --st_device cuda
```
- Outputs:
  - `per_utterance.csv`, `models.csv`, `models_with_baseline.csv`
  - `top_confusions.csv`, `taxonomy.json`, `disagreement.csv`
  - `summary.html`, `summary.json`

## Semantic similarity
- Uses `sentence-transformers` (defaults to `paraphrase-multilingual-MiniLM-L12-v2`).
- Device: `--st_device {cpu|cuda}`; adjust batch `--st_bs` if needed.
- First run downloads to `transcribe/.hf`.
