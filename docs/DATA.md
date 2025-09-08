# Datasets & Manifests

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

## Paths
- Train: `data/train.jsonl` (N≈2047)
- Val: `data/val.jsonl`

## Notes
- Some existing references contain mojibake (encoding artifacts). Prefer fixing source and re-exporting JSONL before fine‑tuning.
