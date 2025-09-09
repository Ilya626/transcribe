#!/usr/bin/env python
"""
Extract a small subset of an HF dataset to local WAV files and build a JSONL manifest.

This avoids relying on dataset-internal paths and ensures local, stable audio paths.

Example:
  python transcribe/tools/extract_hf_audio.py \
    --dataset hf-internal-testing/librispeech_asr_dummy --split validation \
    --audio_col audio --text_col text --max_total 16 \
    --out_dir data/hf_dump/librispeech_dummy --out_manifest data/dummy_local16.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import soundfile as sf
from datasets import load_dataset


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--audio_col", default="audio")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_total", type=int, default=32)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--source_lang", default="ru")
    ap.add_argument("--target_lang", default="ru")
    ap.add_argument("--pnc", default="yes", choices=["yes", "no"])
    args = ap.parse_args()

    configure_local_caches()

    ds = load_dataset(args.dataset, args.config, split=args.split, trust_remote_code=bool(args.trust_remote_code))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_manifest.open("w", encoding="utf-8") as fo:
        for i, ex in enumerate(ds):
            a = ex.get(args.audio_col)
            if not isinstance(a, dict):
                continue
            arr = a.get("array")
            sr = a.get("sampling_rate")
            if arr is None or sr is None:
                continue
            txt = str(ex.get(args.text_col, ""))
            # Write WAV
            wav_path = out_dir / f"utt_{i:06d}.wav"
            sf.write(str(wav_path), arr, sr)
            row = {
                "audio_filepath": str(wav_path.resolve()),
                "text": txt,
                "source_lang": args.source_lang,
                "target_lang": args.target_lang,
                "pnc": args.pnc,
            }
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.max_total and n >= args.max_total:
                break
    print(f"[OK] Dumped {n} files -> {out_dir}; manifest -> {out_manifest}")


if __name__ == "__main__":
    main()

