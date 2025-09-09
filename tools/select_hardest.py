#!/usr/bin/env python
"""
Select the "hardest" utterances for fine-tuning from references and predictions.

Pipeline:
 1) Ensure advanced analysis CSV exists (per_utterance.csv). If not provided,
    run: python -m transcribe.evaluation.analyze_errors <ref> --pred <mid>=<pred> --outdir <analysis_dir> [--no_semantic]
 2) Load per_utterance.csv, rank by WER (desc), tie-break by CER (desc) and sim (asc).
 3) Optional: stratify by a group key present in reference JSONL rows (e.g., dataset)
 4) Optionally mix in a portion of "easy" items and optional non-speech rows
 5) Emit JSONL with selected rows, preserving fields from reference rows; add field "difficulty": hard|easy|nospeech

Usage example:
  python transcribe/tools/select_hardest.py \
    --ref data/ru_megamix_concat.jsonl \
    --pred transcribe/preds/canary_megamix.json \
    --out data/ru_megamix_hard25_easy10_ns3.jsonl \
    --bottom_q 0.25 --easy_ratio 0.10 \
    --nonspeech data/nonspeech.jsonl --nonspeech_ratio 0.03 \
    --group_key dataset --analysis_dir transcribe/preds/analysis_megamix --no_semantic
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def load_jsonl_map(path: Path) -> Dict[str, dict]:
    """Map audio_filepath -> row (keeps all fields)."""
    mp: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            k = o.get("audio_filepath") or o.get("audio")
            if k:
                mp[str(k)] = o
    return mp


def run_analysis_if_missing(ref: Path, pred: Path, model_id: str, outdir: Path, no_semantic: bool) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    per_csv = outdir / "per_utterance.csv"
    if per_csv.exists():
        return per_csv
    import subprocess, sys
    cmd = [sys.executable, "-m", "transcribe.evaluation.analyze_errors", str(ref), "--pred", f"{model_id}={pred}", "--outdir", str(outdir)]
    if no_semantic:
        cmd.append("--no_semantic")
    print("[ANALYZE] ", " ".join(cmd))
    subprocess.check_call(cmd)
    if not per_csv.exists():
        raise SystemExit(f"Analysis did not produce {per_csv}")
    return per_csv


def load_per_utt_csv(path: Path, model_id: str) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if str(row.get("model_id", "")).strip() != model_id:
                continue
            try:
                rows.append({
                    "utt_id": row.get("utt_id"),
                    "wer": float(row.get("wer", 0.0) or 0.0),
                    "cer": float(row.get("cer", 0.0) or 0.0),
                    "ser": float(row.get("ser", 0.0) or 0.0),
                    "sim": None if row.get("sim") in (None, "", "None") else float(row.get("sim")),
                })
            except Exception:
                continue
    return rows


def stratified_bottom_quantile(
    scored: List[dict],
    bottom_q: float,
    group_of: Optional[Dict[str, str]] = None,
    group_key_name: str = "dataset",
) -> List[str]:
    """Return list of utt_ids in the bottom quantile of WER.

    If group_of is provided (utt_id -> group name), the selection is done per
    group to avoid overfitting to a single dataset.
    """
    bottom_q = max(0.0, min(1.0, bottom_q))
    if not scored:
        return []
    # Sort by (wer desc, cer desc, sim asc)
    scored_sorted = sorted(scored, key=lambda x: (-(x.get("wer", 0.0)), -(x.get("cer", 0.0)), (x.get("sim") or 0.0)))

    if group_of is None:
        n = len(scored_sorted)
        k = int(math.ceil(n * bottom_q))
        return [s["utt_id"] for s in scored_sorted[:k]]

    # Stratified per group
    by_group: Dict[str, List[dict]] = {}
    for s in scored_sorted:
        g = group_of.get(s["utt_id"], "")
        by_group.setdefault(g, []).append(s)
    selected: List[str] = []
    for g, lst in by_group.items():
        n = len(lst)
        k = int(math.ceil(n * bottom_q))
        selected.extend([s["utt_id"] for s in lst[:k]])
    return selected


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="Reference JSONL/JSON with audio_filepath/text")
    ap.add_argument("--pred", required=True, help="Predictions JSON (keys match audio_filepath)")
    ap.add_argument("--out", required=True, help="Output JSONL selection")
    ap.add_argument("--model_id", default="canary", help="Model id label used in analysis (default: canary)")
    ap.add_argument("--analysis_dir", default=None, help="Directory with per_utterance.csv; if missing, script will run analysis")
    ap.add_argument("--no_semantic", action="store_true", help="Skip semantic similarity in analysis for speed")
    ap.add_argument("--bottom_q", type=float, default=0.25, help="Bottom quantile by WER to select as hard")
    ap.add_argument("--easy_ratio", type=float, default=0.15, help="Portion of easy examples to mix relative to final size (0.15 => 15%)")
    ap.add_argument("--nonspeech", default=None, help="Optional non-speech manifest JSONL")
    ap.add_argument("--nonspeech_ratio", type=float, default=0.03, help="Portion of non-speech to mix relative to final size")
    ap.add_argument("--group_key", default=None, help="Optional key in ref JSONL rows for stratification (e.g., dataset)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ref_path = Path(args.ref)
    pred_path = Path(args.pred)
    out_path = Path(args.out)

    # Prepare analysis CSV
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else out_path.with_suffix("").parent / (out_path.stem + "_analysis")
    per_csv = run_analysis_if_missing(ref_path, pred_path, args.model_id, analysis_dir, no_semantic=args.no_semantic)

    # Load scored rows for the specified model
    scored = load_per_utt_csv(per_csv, args.model_id)

    # Load reference rows to preserve fields and optional group key
    ref_map = load_jsonl_map(ref_path) if ref_path.suffix.lower() == ".jsonl" else {}
    if not ref_map:
        # Load JSON mapping if not JSONL
        try:
            obj = json.loads(Path(ref_path).read_text(encoding="utf-8"))
            ref_map = {k: {"audio_filepath": k, "text": v} for k, v in obj.items() if isinstance(v, str)}
        except Exception:
            ref_map = {}

    group_of: Optional[Dict[str, str]] = None
    if args.group_key:
        group_of = {k: str(v.get(args.group_key, "")) for k, v in ref_map.items()}

    # Compute stratified bottom quantile selection
    hard_ids = set(str(uid) for uid in stratified_bottom_quantile(scored, bottom_q=float(args.bottom_q), group_of=group_of))

    # Build final list: start with hard
    rng = random.Random(int(args.seed))
    final_rows: List[dict] = []
    for s in scored:
        uid = str(s["utt_id"]) if s.get("utt_id") else None
        if uid in hard_ids:
            base = ref_map.get(uid, {"audio_filepath": uid, "text": None})
            row = dict(base)
            row["difficulty"] = "hard"
            row["wer"] = float(s.get("wer", 0.0))
            row["cer"] = float(s.get("cer", 0.0))
            final_rows.append(row)

    # Optionally add easy portion (from non-hard, lowest WER)
    if args.easy_ratio and args.easy_ratio > 0:
        target_easy = int(round(len(final_rows) * float(args.easy_ratio)))
        candidates = [s for s in scored if str(s.get("utt_id")) not in hard_ids]
        easy_sorted = sorted(candidates, key=lambda x: (float(x.get("wer", 0.0)), float(x.get("cer", 0.0)), -float(x.get("sim")) if x.get("sim") is not None else 0.0))
        for s in easy_sorted[:target_easy]:
            uid = str(s["utt_id"]) if s.get("utt_id") else None
            base = ref_map.get(uid, {"audio_filepath": uid, "text": None})
            row = dict(base)
            row["difficulty"] = "easy"
            row["wer"] = float(s.get("wer", 0.0))
            row["cer"] = float(s.get("cer", 0.0))
            final_rows.append(row)

    # Optional non-speech mix
    if args.nonspeech and args.nonspeech_ratio and args.nonspeech_ratio > 0:
        ns_path = Path(args.nonspeech)
        if ns_path.exists():
            ns_rows = list(load_jsonl_map(ns_path).values())
            target_ns = int(round(len(final_rows) * float(args.nonspeech_ratio)))
            rng.shuffle(ns_rows)
            for r in ns_rows[:target_ns]:
                rr = dict(r)
                rr.setdefault("text", "")
                rr["difficulty"] = "nospeech"
                final_rows.append(rr)
        else:
            print(f"[WARN] Non-speech manifest not found: {ns_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out_path, final_rows)
    print(f"[OK] Wrote selection: {n} rows -> {out_path}")


if __name__ == "__main__":
    main()

