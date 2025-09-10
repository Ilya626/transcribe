# Datasets & Manifests
[← Back to Documentation Index](README.md)

See [Setup & Environment](ENV.md) for configuring caches. All examples below
assume the current working directory is the repository root (`transcribe`).
Set `HF_TOKEN` in the environment if a dataset requires authentication.

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
# PowerShell (Windows)
. env.ps1
# or bash (Linux/Runpod)
source env.sh

# Common Voice v17 (ru)
python tools/build_manifest_hf.py --preset cv17-ru --out data/cv17_ru.jsonl --drop_empty

# Multilingual LibriSpeech (ru)
python tools/build_manifest_hf.py --preset mls-ru --out data/mls_ru.jsonl --drop_empty

# FLEURS (ru)
python tools/build_manifest_hf.py --preset fleurs-ru --out data/fleurs_ru.jsonl --drop_empty

# Russian LibriSpeech (mirror may vary; script tries a few IDs)
python tools/build_manifest_hf.py --preset ruls --out data/ruls.jsonl --drop_empty

# GOLOS (crowd/farfield; mirrors/configs may vary)
python tools/build_manifest_hf.py --preset golos-crowd --out data/golos_crowd.jsonl --drop_empty
python tools/build_manifest_hf.py --preset golos-farfield --out data/golos_farfield.jsonl --drop_empty

# Podlodka Speech (Russian podcasts)
python tools/build_manifest_hf.py --preset podlodka --out data/podlodka.jsonl --drop_empty

# Telephony (UniDataPro; may require auth)
python tools/build_manifest_hf.py \
  --dataset UniDataPro/russian-speech-recognition-dataset \
  --split train+validation+test --audio_col audio --text_col transcript \
  --out data/telephony.jsonl --drop_empty

# Optional non-speech (for anti-hallucination; emits empty text and nospeech=true)
python tools/build_manifest_hf.py --preset audioset-nonspeech --out data/nonspeech.jsonl

# Mix manifests (concatenate)
python tools/mix_manifests.py \
  --in cv=data/cv17_ru.jsonl --in mls=data/mls_ru.jsonl --in fleurs=data/fleurs_ru.jsonl \
  --in ruls=data/ruls.jsonl --in golos_c=data/golos_crowd.jsonl --in golos_f=data/golos_farfield.jsonl \
  --out data/ru_megamix_concat.jsonl --add_dataset_tag

# Or sample to a target size with ratios
python tools/mix_manifests.py \
  --in cv=data/cv17_ru.jsonl --in mls=data/mls_ru.jsonl --in fleurs=data/fleurs_ru.jsonl \
  --in ruls=data/ruls.jsonl --in golos_c=data/golos_crowd.jsonl --in golos_f=data/golos_farfield.jsonl \
  --out data/ru_megamix_500k.jsonl --target_size 500000 \
  --ratios cv=0.35,mls=0.2,fleurs=0.05,ruls=0.15,golos_c=0.15,golos_f=0.1 \
  --shuffle --seed 42 --add_dataset_tag
```

Here `golos_mix.jsonl` is Golos crowd+farfield concatenated.

### Canary quality filter
Run Canary on a manifest and keep only high‑quality rows (WER ≤ 0.15 by default):

```bash
python tools/filter_manifest_canary.py \
   --manifest data/cv17_ru.jsonl --out data/cv17_ru_sel.jsonl \
   --max_wer 0.15 --min_dur 1 --max_dur 35 --batch_size 64
```

Requires `HF_TOKEN` with access to `nvidia/canary-1b-v2` (downloads its `.nemo`).

Repeat for each dataset before mixing.

### Stage-1 RU mix (methodology ratios)
After generating the individual manifests above, combine them using the methodology default ratios:

```bash
python tools/build_ru_stage1_mix.py \
  --golos data/golos_mix.jsonl --cv data/cv17_ru.jsonl \
  --ruls data/ruls.jsonl --podlodka data/podlodka.jsonl \
  --telephony data/telephony.jsonl --nonspeech data/nonspeech.jsonl \
  --out data/ru_stage1_mix.jsonl --target_size 1000000 \
  --shuffle --seed 42 --add_dataset_tag
```

The script uses Golos 35%, Common Voice 25%, RuLibriSpeech 15%, Podlodka 10%, Telephony 10% and non-speech 5%.
Use `--ratios` to override.

### Stage-1 pipeline helper
Fetch HF datasets, convert them to manifests, filter with Canary and build the Stage‑1 mix in one go:

```bash
python tools/build_ru_stage1_pipeline.py \
  --telephony data/telephony.jsonl --out_dir data \
  --target_size 1000000 --include_nonspeech --seed 42 --batch_size 64
```

The helper downloads Common Voice, Russian LibriSpeech, Podlodka and both GOLOS
splits, runs `filter_manifest_canary` on each plus your telephony manifest,
concatenates GOLOS crowd+farfield and finally produces `ru_stage1_mix.jsonl`.
Provide the telephony manifest manually.

For NVIDIA A6000 GPUs (48 GB VRAM) Canary runs reliably with batches of around
64 samples and can sometimes handle 128. Adjust `--batch_size` accordingly.

Tips
- If a preset fails (HF ID changed), specify `--dataset/--config/--split/--audio_col/--text_col` explicitly.
- All emitted rows include `source_lang/target_lang/pnc` (defaults ru/ru/yes) so they plug into the NeMo training scripts.
- For non-speech, integrate conservatively (e.g., 2–5%) and monitor stability; consider keeping it in a separate manifest until validated.

## Convert local HF cache to manifest

If a dataset has already been downloaded to a Hugging Face cache (`.hf/datasets/.../<hash>`), turn it into `train/validation/test` JSONL manifests with:

```bash
python tools/hf_cache_to_manifest.py \
  --dataset_dir .hf/datasets/bond005___rulibrispeech/default/0.0.0/<hash> \
  --out_dir data/rulibrispeech
```

The tool descends into split subdirectories and merges multiple `.arrow` shards per split. Audio/text column names are auto‑detected but can be overridden via `--audio_col`/`--text_col`.

Next: [Inference Runbook](RUNBOOK.md)
