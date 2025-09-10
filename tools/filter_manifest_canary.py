#!/usr/bin/env python
"""Transcribe a manifest with Canary and keep rows meeting quality thresholds."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import soundfile as sf
from jiwer import wer


def run(cmd: List[str], cwd: Path) -> None:
    """Run a subprocess in ``cwd`` echoing the command.

    Ensures the parent of ``cwd`` is on ``PYTHONPATH`` so the ``transcribe``
    package is importable when invoking modules via ``-m``.
    """
    print("+", " ".join(cmd))
    env = dict(os.environ)
    parent = str(cwd.parent)
    env["PYTHONPATH"] = (
        parent + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def transcribe_if_needed(
    manifest: Path,
    pred_path: Path,
    model_id: str,
    batch_size: int | None,
    repo_root: Path,
) -> None:
    if pred_path.exists():
        return
    cmd = [
        sys.executable,
        "-m",
        "transcribe.models.inference_canary_nemo",
        str(manifest),
        str(pred_path),
        "--model_id",
        model_id,
    ]
    if batch_size:
        cmd.extend(["--batch_size", str(batch_size)])
    run(cmd, cwd=repo_root)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


def write_jsonl(path: Path, rows: List[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Input JSONL manifest with audio_filepath/text")
    ap.add_argument("--out", required=True, help="Output JSONL path for filtered rows")
    ap.add_argument("--pred", default=None, help="Optional existing predictions JSON; if missing, will run Canary")
    ap.add_argument("--model_id", default="nvidia/canary-1b-v2", help="HF model id for Canary")
    ap.add_argument("--batch_size", type=int, default=None, help="Batch size for Canary inference")
    ap.add_argument("--max_wer", type=float, default=0.15, help="Maximum WER to keep a sample")
    ap.add_argument("--min_dur", type=float, default=1.0, help="Minimum duration in seconds")
    ap.add_argument("--max_dur", type=float, default=35.0, help="Maximum duration in seconds")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    pred_path = Path(args.pred) if args.pred else out_path.with_suffix(".pred.json")

    repo_root = Path(__file__).resolve().parents[1]
    transcribe_if_needed(manifest_path, pred_path, args.model_id, args.batch_size, repo_root)

    try:
        preds: Dict[str, str] = json.loads(pred_path.read_text(encoding="utf-8"))
    except Exception:
        preds = {}

    selected: List[dict] = []
    for row in iter_jsonl(manifest_path):
        audio = str(row.get("audio_filepath") or row.get("audio"))
        ref = str(row.get("text") or "")
        if not audio or not ref:
            continue
        try:
            dur = float(sf.info(audio).duration)
        except Exception:
            try:
                data, sr = sf.read(audio)
                dur = len(data) / float(sr)
            except Exception:
                continue
        if dur < float(args.min_dur) or dur > float(args.max_dur):
            continue
        pred = preds.get(audio, "")
        if not pred:
            continue
        w = wer(ref, pred)
        if w <= float(args.max_wer):
            out_row = dict(row)
            out_row["pred_text"] = pred
            out_row["wer"] = float(w)
            selected.append(out_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out_path, selected)
    print(f"[OK] Selected {n} rows -> {out_path}")


if __name__ == "__main__":
    main()
