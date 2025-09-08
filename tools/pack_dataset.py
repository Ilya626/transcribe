#!/usr/bin/env python
"""
Pack a JSONL manifest into a portable form:
- Copy all referenced audio files into a single target folder (default: data/Train data)
- Rewrite the JSONL so that `audio_filepath` points to relative, POSIX-style paths

Usage (Linux):
  python transcribe/tools/pack_dataset.py \
    --input data/train.jsonl \
    --out_dir "data/Train data" \
    --out_json data/train_portable.jsonl

Usage (Windows PowerShell):
  . transcribe/env.ps1; .\\transcribe\\.venv\\Scripts\\python \\
    transcribe\\tools\\pack_dataset.py \\
    --input data\\train.jsonl \\
    --out_dir "data\\Train data" \\
    --out_json data\\train_portable.jsonl

Notes:
- Filenames are made unique by appending an index if collisions occur.
- Paths in the output JSONL are relative to the output JSONL parent, e.g. "Train data/clip_0001.wav".
- Output uses forward slashes for cross-platform portability.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def posix_relpath(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
    except Exception:
        rel = path
    return rel.as_posix()


def unique_name(dst_dir: Path, stem: str, ext: str, taken: set[str]) -> str:
    name = f"{stem}{ext}"
    if name not in taken and not (dst_dir / name).exists():
        return name
    i = 1
    while True:
        cand = f"{stem}_{i:04d}{ext}"
        if cand not in taken and not (dst_dir / cand).exists():
            return cand
        i += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Source JSONL with audio_filepath/text")
    ap.add_argument("--out_dir", default="data/Train data", help="Target folder to copy all audio into")
    ap.add_argument(
        "--out_json",
        default=None,
        help="Output JSONL (default: alongside input, named *_portable.jsonl)",
    )
    ap.add_argument("--dry_run", action="store_true", help="Do not copy files; just print plan")
    args = ap.parse_args()

    src_json = Path(args.input)
    if not src_json.exists():
        raise SystemExit(f"Input not found: {src_json}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_json is None:
        out_json = src_json.with_name(src_json.stem + "_portable.jsonl")
    else:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)

    # We’ll make output paths relative to the directory containing out_json
    out_base = out_json.parent.resolve()

    # If out_dir is not under out_json parent, compute a relative that still works
    # For simplicity, we will point to relative path from out_json parent -> out_dir
    rel_out_dir = Path(os.path.relpath(out_dir.resolve(), out_base)).as_posix()

    taken: set[str] = set()
    copied = 0
    missing = 0

    with open(src_json, "r", encoding="utf-8-sig") as fin, open(
        out_json, "w", encoding="utf-8"
    ) as fout:
        for ln in fin:
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                # skip bad line
                continue
            src = obj.get("audio_filepath") or obj.get("audio")
            if not src:
                continue
            src_path = Path(src)
            # Accept both backslashes and forward slashes
            if not src_path.exists():
                src_path = Path(str(src).replace("\\", "/"))
            if not src_path.exists():
                # Try absolute expanduser
                src_path = Path(os.path.expanduser(str(src_path)))
            if not src_path.exists():
                missing += 1
                continue

            stem = src_path.stem
            ext = src_path.suffix
            dst_name = unique_name(out_dir, stem, ext, taken)
            taken.add(dst_name)
            dst_path = out_dir / dst_name
            if not args.dry_run:
                shutil.copy2(src_path, dst_path)
            copied += 1

            # Rewrite path relative to out_json parent dir, using POSIX slashes
            rel_path = f"{rel_out_dir}/{dst_name}" if rel_out_dir != "." else dst_name
            obj["audio_filepath"] = rel_path
            # Keep text and other fields as is; remove any alternate key "audio"
            if "audio" in obj:
                del obj["audio"]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        f"Done. Copied={copied}, missing={missing}. Output JSONL -> {out_json}  Audio dir -> {out_dir}"
    )


if __name__ == "__main__":
    main()

