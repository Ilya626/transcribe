#!/usr/bin/env python
"""
Build a JSONL manifest (audio_filepath, text) from a Hugging Face dataset.

Supports presets for common RU corpora (Common Voice RU, MLS RU, FLEURS RU,
RuLS, GOLOS crowd/farfield, Podlodka, optional AudioSet non-speech).

Examples:

  # Common Voice v17 RU (all splits), limit to 200k rows
  python transcribe/tools/build_manifest_hf.py \
    --preset cv17-ru --out data/cv17_ru.jsonl --max_total 200000

  # Explicit dataset/config/split and columns
  python transcribe/tools/build_manifest_hf.py \
    --dataset mozilla-foundation/common_voice_17_0 --config ru --split train+validation \
    --audio_col audio --text_col sentence --out data/cv17_ru_tval.jsonl

  # Add language/task fields
  python transcribe/tools/build_manifest_hf.py --preset mls-ru --out data/mls_ru.jsonl \
    --source_lang ru --target_lang ru --task asr --pnc yes

Notes:
 - Audio files are not copied; we reference HF cache paths. Ensure your env
   sets HF_HOME inside the repo (use transcribe/env.ps1 or env.sh) to keep
   everything local.
 - For optional non-speech, we emit empty text ("") and add custom flag
   {"nospeech": true}. Integrate with care in training.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import Audio, load_dataset, concatenate_datasets


def configure_local_caches() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    hf = os.environ.get("HF_HOME") or str(repo_root / ".hf")
    os.environ.setdefault("HF_HOME", hf)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(hf) / "hub"))
    os.environ.setdefault("TORCH_HOME", str(repo_root / ".torch"))
    tmp = str(repo_root / ".tmp")
    os.environ.setdefault("TMP", tmp)
    os.environ.setdefault("TEMP", tmp)
    for d in [hf, os.environ["HF_HUB_CACHE"], os.environ["TORCH_HOME"], tmp]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _load_with_candidates(cands, audio_col: str | None, text_col: str | None, split: str | None, trust_remote_code: bool = False):
    last_err: Optional[Exception] = None
    for c in cands:
        ds_id = c.dataset
        cfg = c.config
        sp = split or c.split
        a_col = audio_col or c.audio_col
        t_col = text_col or c.text_col
        try:
            parts = [s.strip() for s in sp.split("+") if s.strip()]
            datasets = []
            for p in parts:
                # Avoid trust_remote_code for compatibility with datasets>=3
                ds = load_dataset(ds_id, cfg, split=p, trust_remote_code=trust_remote_code)
                if a_col in ds.column_names:
                    ds = ds.cast_column(a_col, Audio(decode=False))
                datasets.append(ds)
            ds = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
            return ds, a_col, t_col, ds_id, cfg, sp
        except Exception as e:  # try next candidate
            last_err = e
            continue
    raise SystemExit(f"Failed to load any candidate. Last error: {last_err}")


def iter_rows(ds, audio_col: str, text_col: str, drop_empty: bool, path_col: Optional[str] = "file") -> Iterable[dict]:
    nospeech_flag = False
    # Heuristic: if the text_col is missing in features, we may be in a
    # non-speech preset; in this case we emit empty text.
    if text_col not in ds.column_names:
        nospeech_flag = True
    for ex in ds:
        # HF Audio feature stores a record; when cast with decode=False, we get
        # {'path': <local_path>, 'array': None, 'sampling_rate': ...}.
        a = ex.get(audio_col)
        path = None
        # Prefer explicit path column if present (e.g., 'file' in LibriSpeech dummy)
        if path_col and path_col in ex and isinstance(ex[path_col], str) and ex[path_col]:
            path = ex[path_col]
        elif isinstance(a, dict):
            path = a.get("path")
        elif isinstance(a, str):
            path = a
        if not path:
            continue
        txt = ex.get(text_col) if not nospeech_flag else ""
        if not isinstance(txt, str):
            # Some datasets store text under different names; skip if not str.
            continue
        txt = txt.strip() if txt else ""
        if drop_empty and not txt:
            # For non-speech sets, keep empty if explicitly requested later.
            if not nospeech_flag:
                continue
        row = {
            "audio_filepath": str(path),
            "text": txt,
        }
        if nospeech_flag:
            row["nospeech"] = True
        yield row


def main() -> None:
    from transcribe.tools.presets_ru_datasets import PRESETS

    ap = argparse.ArgumentParser(description=__doc__)
    gds = ap.add_argument_group("Dataset selection")
    gds.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None)
    gds.add_argument("--dataset", default=None, help="HF dataset id (org/name)")
    gds.add_argument("--config", default=None, help="HF dataset config (e.g., ru, crowd)")
    gds.add_argument("--split", default=None, help="Split or 'a+b+c' to concat (e.g., train+validation)")
    gmap = ap.add_argument_group("Column mapping")
    gmap.add_argument("--audio_col", default=None)
    gmap.add_argument("--path_col", default="file", help="Optional explicit path column to prefer if present (e.g., 'file')")
    gmap.add_argument("--text_col", default=None)
    gout = ap.add_argument_group("Output")
    gout.add_argument("--out", required=True, help="Output JSONL path")
    gout.add_argument("--max_total", type=int, default=None)
    gout.add_argument("--drop_empty", action="store_true", help="Drop rows with empty text (except non-speech presets)")
    gout.add_argument("--source_lang", default="ru")
    gout.add_argument("--target_lang", default="ru")
    gout.add_argument("--task", choices=["asr", "translate"], default="asr")
    gout.add_argument("--pnc", choices=["yes", "no"], default="yes")
    ap.add_argument("--trust_remote_code", action="store_true", help="Allow loading datasets with custom scripts (HF)")
    args = ap.parse_args()

    configure_local_caches()

    if args.preset:
        from transcribe.tools.presets_ru_datasets import PRESETS as _P
        ds, a_col, t_col, ds_id, cfg, sp = _load_with_candidates(_P[args.preset], args.audio_col, args.text_col, args.split, trust_remote_code=bool(args.trust_remote_code))
        print(f"[HF] Loaded preset {args.preset}: id={ds_id} cfg={cfg} split={sp} a={a_col} t={t_col} rows={len(ds)}")
    else:
        if not args.dataset or not args.split:
            raise SystemExit("Provide either --preset or --dataset + --split")
        parts = [s.strip() for s in args.split.split("+") if s.strip()]
        datasets = []
        for p in parts:
            ds = load_dataset(args.dataset, args.config, split=p, trust_remote_code=bool(args.trust_remote_code))
            if args.audio_col and args.audio_col in ds.column_names:
                ds = ds.cast_column(args.audio_col, Audio(decode=False))
            datasets.append(ds)
        ds = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
        a_col = args.audio_col or "audio"
        t_col = args.text_col or "text"
        print(f"[HF] Loaded id={args.dataset} cfg={args.config} split={args.split} rows={len(ds)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fo:
        for row in iter_rows(ds, a_col, t_col, args.drop_empty, path_col=args.path_col):
            row.setdefault("source_lang", args.source_lang)
            row.setdefault("target_lang", args.target_lang)
            row.setdefault("pnc", args.pnc)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.max_total and n >= args.max_total:
                break
    print(f"[OK] Wrote {n} rows -> {out_path}")


if __name__ == "__main__":
    main()
