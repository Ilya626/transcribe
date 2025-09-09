# Datasets & Manifests
[← Back to Documentation Index](README.md)

See [Setup & Environment](ENV.md) for configuring caches.

## Reference formats
- JSONL: one object per line with required fields:
  - `audio_filepath`: absolute or relative path to WAV/FLAC/MP3
  - `text`: reference transcript
  - Optional: `source_lang`, `target_lang`, `pnc`
- JSON: mapping `path -> text`

## Example
```
{"audio_filepath": ".../segments/seg_0001.wav", "text": "..."}
```

## Convert JSONL to JSON references (PowerShell)
```
$in = 'data/train.jsonl'; $out = 'data/train_refs.json'
$m = @{}; Get-Content $in | ForEach-Object { $o = $_ | ConvertFrom-Json; $m[$o.audio_filepath] = $o.text }
$m | ConvertTo-Json -Depth 1 | Set-Content -Encoding UTF8 $out
```

## Paths
- Train: `data/train.jsonl` (N≈2047)
- Val: `data/val.jsonl`

## Notes
- Some existing references contain mojibake (encoding artifacts). Prefer fixing source and re-exporting JSONL before fine‑tuning.

## RU Megamix Prep (HF datasets)

Use the helper to pull datasets from Hugging Face and convert to our JSONL format (audio_filepath, text):

```
# Activate project-local caches (keeps HF audio under transcribe/.hf)
. transcribe/env.ps1   # PowerShell (Windows)
# or: source transcribe/env.sh  # bash (Linux/Runpod)

# Common Voice v17 (ru)
python transcribe/tools/build_manifest_hf.py --preset cv17-ru --out data/cv17_ru.jsonl --drop_empty

# Multilingual LibriSpeech (ru)
python transcribe/tools/build_manifest_hf.py --preset mls-ru --out data/mls_ru.jsonl --drop_empty

# FLEURS (ru)
python transcribe/tools/build_manifest_hf.py --preset fleurs-ru --out data/fleurs_ru.jsonl --drop_empty

# Russian LibriSpeech (mirror may vary; script tries a few IDs)
python transcribe/tools/build_manifest_hf.py --preset ruls --out data/ruls.jsonl --drop_empty

# GOLOS (crowd/farfield; mirrors/configs may vary)
python transcribe/tools/build_manifest_hf.py --preset golos-crowd --out data/golos_crowd.jsonl --drop_empty
python transcribe/tools/build_manifest_hf.py --preset golos-farfield --out data/golos_farfield.jsonl --drop_empty

# Optional non-speech (for anti-hallucination; emits empty text and nospeech=true)
python transcribe/tools/build_manifest_hf.py --preset audioset-nonspeech --out data/nonspeech.jsonl

# Mix manifests (concatenate)
python transcribe/tools/mix_manifests.py \
  --in cv=data/cv17_ru.jsonl --in mls=data/mls_ru.jsonl --in fleurs=data/fleurs_ru.jsonl \
  --in ruls=data/ruls.jsonl --in golos_c=data/golos_crowd.jsonl --in golos_f=data/golos_farfield.jsonl \
  --out data/ru_megamix_concat.jsonl --add_dataset_tag

# Or sample to a target size with ratios
python transcribe/tools/mix_manifests.py \
  --in cv=data/cv17_ru.jsonl --in mls=data/mls_ru.jsonl --in fleurs=data/fleurs_ru.jsonl \
  --in ruls=data/ruls.jsonl --in golos_c=data/golos_crowd.jsonl --in golos_f=data/golos_farfield.jsonl \
  --out data/ru_megamix_500k.jsonl --target_size 500000 \
  --ratios cv=0.35,mls=0.2,fleurs=0.05,ruls=0.15,golos_c=0.15,golos_f=0.1 \
  --shuffle --seed 42 --add_dataset_tag
```

Tips
- If a preset fails (HF ID changed), specify `--dataset/--config/--split/--audio_col/--text_col` explicitly.
- All emitted rows include `source_lang/target_lang/pnc` (defaults ru/ru/yes) so they plug into the NeMo training scripts.
- For non-speech, integrate conservatively (e.g., 2–5%) and monitor stability; consider keeping it in a separate manifest until validated.

## Convert local HF cache to manifest

If a dataset has already been downloaded to a Hugging Face cache (`.hf/datasets/.../<hash>`), turn it into `train/validation/test` JSONL manifests with:

```bash
python transcribe/tools/hf_cache_to_manifest.py \
  --dataset_dir .hf/datasets/bond005___rulibrispeech/default/0.0.0/<hash> \
  --out_dir data/rulibrispeech
```

The tool descends into split subdirectories and merges multiple `.arrow` shards per split. Audio/text column names are auto‑detected but can be overridden via `--audio_col`/`--text_col`.

Next: [Inference Runbook](RUNBOOK.md)
