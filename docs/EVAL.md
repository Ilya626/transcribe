# Evaluation & Analysis
[← Back to Documentation Index](README.md)

See [Inference Runbook](RUNBOOK.md) for generating predictions.

## Basic metrics
- Compare a single predictions file: `python transcribe/evaluation/evaluate_transcriptions.py <ref.json|jsonl> <pred.json>`
- Compare multiple: `python transcribe/evaluation/compare_transcriptions.py reference.json whisper.json canary.json gigaam.json`
- The script prints word error rate (WER), character error rate (CER), sentence error rate (SER) and a semantic similarity score. It also reports counts of substitutions, insertions and deletions.

## Advanced analysis
- Command template:
```
python -m transcribe.evaluation.analyze_errors data/train.jsonl \
  --pred name1=path1.json \
  --pred name2=path2.json \
  --outdir transcribe/preds/analysis_<tag> --st_device cuda --bert_score
```
- Outputs:
  - `per_utterance.csv`, `models.csv`, `models_with_baseline.csv`
  - `top_confusions.csv`, `taxonomy.json`, `disagreement.csv`
  - `summary.html`, `summary.json`, `error_overlap_heatmap.png`
- Options:
  - `--no_semantic`: disable semantic similarity (faster, offline).
  - `--no_ner`: disable named-entity extraction.
  - `--st_model`: sentence-transformers model name (default `paraphrase-multilingual-MiniLM-L12-v2`).
  - `--st_device`: device for sentence-transformers (`cpu` default; set `cuda` to use GPU).
  - `--st_bs`: sentence-transformers batch size (default 256).
  - `--sample N --seed S`: analyze a random subset.
- Evaluation additionally normalizes `ё`→`е` for robust matching.

## Semantic similarity
- Uses `sentence-transformers` (defaults to `paraphrase-multilingual-MiniLM-L12-v2`).
- Device: `--st_device {cpu|cuda}`; adjust batch `--st_bs` if needed.
- First run downloads to `transcribe/.hf`.
- Prefetch once:
  - `python -c "from sentence_transformers import SentenceTransformer as ST; ST('paraphrase-multilingual-MiniLM-L12-v2')"`
- Advanced CLI frees the semantic model at the end; if you run repeatedly in a long-lived process, call `release_semantic_models()` from `transcribe.evaluation.advanced.semantics` to free memory.

## Notes
- References can be JSONL (`{"audio_filepath": ..., "text": ...}`) or JSON mapping `path->text`.
- Predictions are JSON mapping `path->hypothesis`; keys should match `audio_filepath` from references.
- NER uses `natasha` if available, otherwise a light capitalization heuristic for Russian.
- Recommended multi-model advanced evaluation (excluding Distil by default):
```
python -m transcribe.evaluation.analyze_errors data/train.jsonl \
  --pred v3=transcribe/preds/whisper_large_v3_train_beam2_bs8.json \
  --pred turbo=transcribe/preds/whisper_large_v3_turbo_train_beam2_bs10.json \
  --pred ru_antony66=transcribe/preds/ru_whisper_large_v3_antony66_train.json \
  --pred ru_apelsin=transcribe/preds/ru_whisper_large_v3_apelsin_train.json \
  --pred ru_dvislobokov=transcribe/preds/ru_whisper_large_v3_turbo_dvislobokov_train.json \
  --pred ru_bond005=transcribe/preds/ru_whisper_podlodka_turbo_bond005_train.json \
  --outdir transcribe/preds/analysis_whisper_full --st_device cuda
```

Back to [Inference Runbook](RUNBOOK.md)
