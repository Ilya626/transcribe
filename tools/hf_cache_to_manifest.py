#!/usr/bin/env python
"""
Convert a local Hugging Face dataset cache (``dataset_info.json`` +
``*.arrow`` shards) into JSONL manifests with ``audio_filepath`` and
``text`` fields.

Example:
  python transcribe/tools/hf_cache_to_manifest.py \
    --dataset_dir .hf/datasets/bond005___rulibrispeech/default/0.0.0/XXXXX \
    --out_dir data/rulibrispeech

``dataset_dir`` should point to the leaf cache directory (hash) produced
by `datasets`.  The script descends into split subdirectories (``train``/
``validation``/``test``) and handles multiple ``.arrow`` shards per
split.  Column names for audio/text are auto‑detected but can be
overridden via ``--audio_col``/``--text_col``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import Audio, Dataset


def detect_columns(ds: Dataset, audio_col: str | None, text_col: str | None) -> tuple[str, str]:
    cols = ds.column_names
    a = audio_col or next((c for c in ("audio", "path", "file", "audio_filepath") if c in cols), None)
    t = text_col or next((c for c in ("text", "sentence", "transcription", "normalized_text") if c in cols), None)
    if not a or not t:
        raise SystemExit(
            f"Cannot determine audio/text columns from {cols}; specify --audio_col/--text_col"
        )
    return a, t


def iter_arrow(file: Path, audio_col: str, text_col: str) -> Iterable[dict]:
    ds = Dataset.from_file(str(file))
    if audio_col in ds.column_names:
        ds = ds.cast_column(audio_col, Audio(decode=False))
    for ex in ds:
        a = ex.get(audio_col)
        path = None
        if isinstance(a, dict):
            path = a.get("path") or a.get("file")
        elif isinstance(a, str):
            path = a
        if not path:
            continue
        txt = ex.get(text_col, "")
        if not isinstance(txt, str):
            continue
        yield {"audio_filepath": str(path), "text": txt.strip()}


def guess_split(path: Path) -> str | None:
    """Infer split name (train/validation/test) from path components."""
    parts = [p.lower() for p in path.parts]
    for p in reversed(parts):
        if "train" in p:
            return "train"
        if any(tag in p for tag in ("validation", "valid", "dev")):
            return "validation"
        if "test" in p:
            return "test"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_dir", required=True, help="Path to HF dataset cache directory")
    ap.add_argument("--out_dir", default="data", help="Directory to write output JSONL files")
    ap.add_argument("--audio_col", default=None, help="Override audio column name")
    ap.add_argument("--text_col", default=None, help="Override text column name")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    if not root.exists():
        raise SystemExit(f"Dataset directory not found: {root}")
    if not (root / "dataset_info.json").exists():
        subs = [d for d in root.iterdir() if d.is_dir()]
        if len(subs) == 1 and (subs[0] / "dataset_info.json").exists():
            root = subs[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrows = list(root.rglob("*.arrow"))
    if not arrows:
        raise SystemExit(f"No .arrow files found under {root}")

    split_map: dict[str, list[Path]] = {}
    for f in arrows:
        split = guess_split(f.relative_to(root))
        if split:
            split_map.setdefault(split, []).append(f)

    if not split_map:
        raise SystemExit("Could not infer splits from .arrow paths")

    for split, files in split_map.items():
        files = sorted(files)
        # detect columns using first shard
        first_ds = Dataset.from_file(str(files[0]))
        audio_col, text_col = detect_columns(first_ds, args.audio_col, args.text_col)
        first_ds = first_ds.cast_column(audio_col, Audio(decode=False))

        out_path = out_dir / f"{split}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as fo:
            for ex in first_ds:
                fo.write(json.dumps({"audio_filepath": str(ex[audio_col]["path"] if isinstance(ex[audio_col], dict) else ex[audio_col]),
                                     "text": (ex.get(text_col) or "").strip()}, ensure_ascii=False) + "\n")
                count += 1
            for shard in files[1:]:
                for row in iter_arrow(shard, audio_col, text_col):
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
        print(f"[OK] {split}: wrote {count} rows -> {out_path}")


if __name__ == "__main__":
    main()
