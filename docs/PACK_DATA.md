# Packing Dataset for Portable Paths

Goal: copy all referenced audio into a single folder and rewrite JSONL to use relative, POSIX paths, so it runs the same on Windows and Runpod/Linux.

## Script
- `transcribe/tools/pack_dataset.py`

## Example (Linux / Runpod)
```
source transcribe/env.sh
python transcribe/tools/pack_dataset.py \
  --input data/train.jsonl \
  --out_dir "data/Train data" \
  --out_json data/train_portable.jsonl
```

## Example (Windows PowerShell)
```
. transcribe/env.ps1
.\transcribe\.venv\Scripts\python transcribe\tools\pack_dataset.py \
  --input data\train.jsonl \
  --out_dir "data\Train data" \
  --out_json data\train_portable.jsonl
```

## Usage Notes
- Output JSONL references files as `Train data/<file>` relative to the JSONL location, using forward slashes.
- Colliding basenames are disambiguated by appending `_0001`, `_0002`, etc.
- Non-existent source paths are skipped and reported as `missing` in the script summary.
- You may repeat for validation: `--input data/val.jsonl --out_json data/val_portable.jsonl`.
- Keep the new folder out of Git if it contains large media (add `data/Train data/` to `.gitignore` if needed).
