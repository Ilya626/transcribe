#!/usr/bin/env python
"""
Build the RU Stage‑1 training mix (Golos, CV, RuLibriSpeech, Podlodka,
telephony, optional non‑speech) using methodology ratios.

Each input is a JSONL manifest with at least ``audio_filepath`` and ``text``
fields. Sampling is by number of rows; ratios sum to 1.0 and are applied to the
requested ``--target_size``.

Example:
  python transcribe/tools/build_ru_stage1_mix.py \
    --golos data/golos_mix.jsonl --cv data/cv17_ru.jsonl \
    --ruls data/ruls.jsonl --podlodka data/podlodka.jsonl \
    --telephony data/telephony.jsonl --nonspeech data/nonspeech.jsonl \
    --out data/ru_stage1_mix.jsonl --target_size 1000000 \
    --shuffle --seed 42 --add_dataset_tag
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from transcribe.tools.mix_manifests import reservoir_sample

# Default Stage‑1 ratios from methodology (sum to 1.0)
DEFAULT_RATIOS: Dict[str, float] = {
    "golos": 0.35,
    "cv": 0.25,
    "ruls": 0.15,
    "podlodka": 0.10,
    "telephony": 0.10,
    "nonspeech": 0.05,
}


def parse_ratio_overrides(spec: str) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for part in spec.split(","):
        if not part:
            continue
        name, sval = part.split("=", 1)
        ratios[name.strip()] = float(sval)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Sum of ratios must be > 0")
    for k in list(ratios.keys()):
        ratios[k] = ratios[k] / total
    return ratios


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golos", required=True, help="Golos manifest (crowd+farfield)")
    ap.add_argument("--cv", required=True, help="Common Voice v17 (ru) manifest")
    ap.add_argument("--ruls", required=True, help="Russian LibriSpeech manifest")
    ap.add_argument("--podlodka", required=True, help="Podlodka speech manifest")
    ap.add_argument("--telephony", required=True, help="UniDataPro telephony manifest")
    ap.add_argument("--nonspeech", default=None, help="Optional AudioSet non‑speech manifest")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--target_size", type=int, required=True, help="Total rows in output mix")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle final mix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--add_dataset_tag", action="store_true", help="Add dataset field with source name")
    ap.add_argument("--ratios", default=None, help="Override ratios name=val,...")
    args = ap.parse_args()

    pairs: List[Tuple[str, Path]] = [
        ("golos", Path(args.golos)),
        ("cv", Path(args.cv)),
        ("ruls", Path(args.ruls)),
        ("podlodka", Path(args.podlodka)),
        ("telephony", Path(args.telephony)),
    ]
    if args.nonspeech:
        pairs.append(("nonspeech", Path(args.nonspeech)))

    ratios: Dict[str, float]
    if args.ratios:
        ratios = parse_ratio_overrides(args.ratios)
    else:
        ratios = {nm: DEFAULT_RATIOS[nm] for nm, _ in pairs}
        total = sum(ratios.values())
        ratios = {k: v / total for k, v in ratios.items()}

    target = int(args.target_size)
    plan = {nm: int(round(ratios[nm] * target)) for nm, _ in pairs}
    diff = target - sum(plan.values())
    for nm, _ in pairs:
        if diff == 0:
            break
        plan[nm] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1

    rng = random.Random(args.seed)
    rows: List[dict] = []
    for nm, p in pairs:
        k = max(0, plan[nm])
        if k == 0:
            continue
        sample = reservoir_sample(p, k, seed=args.seed + hash(nm) % 997)
        if args.add_dataset_tag:
            for r in sample:
                r.setdefault("dataset", nm)
        rows.extend(sample)

    if args.shuffle:
        rng.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fo:
        for r in rows[:target]:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Sampled -> {out_path} rows={min(len(rows), target)} plan={plan}")


if __name__ == "__main__":
    main()
