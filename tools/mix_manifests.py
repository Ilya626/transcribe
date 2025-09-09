#!/usr/bin/env python
"""
Mix multiple JSONL manifests into one, with optional sampling ratios.

Usage:
  # Concatenate manifests in order (fast, streaming)
  python transcribe/tools/mix_manifests.py \
    --in cv=data/cv17_ru.jsonl --in mls=data/mls_ru.jsonl \
    --out data/mixed_concat.jsonl --add_dataset_tag

  # Sample to target size with ratios and shuffle (reservoir sampling)
  python transcribe/tools/mix_manifests.py \
    --in cv=data/cv17_ru.jsonl --in mls=data/mls_ru.jsonl --in fleurs=data/fleurs_ru.jsonl \
    --out data/mixed_300k.jsonl --target_size 300000 --ratios cv=0.5,mls=0.3,fleurs=0.2 \
    --shuffle --seed 42 --add_dataset_tag
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_in_args(items: List[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for it in items:
        if "=" in it:
            name, p = it.split("=", 1)
        else:
            p = it
            name = Path(p).stem
        out.append((name, Path(p)))
    return out


def parse_ratios(spec: str) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for part in spec.split(","):
        if not part:
            continue
        name, sval = part.split("=", 1)
        ratios[name.strip()] = float(sval)
    s = sum(ratios.values())
    if s <= 0:
        raise ValueError("Sum of ratios must be > 0")
    for k in list(ratios.keys()):
        ratios[k] = ratios[k] / s
    return ratios


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


def reservoir_sample(path: Path, k: int, seed: int) -> List[dict]:
    rnd = random.Random(seed)
    sample: List[dict] = []
    i = 0
    for row in iter_jsonl(path):
        i += 1
        if len(sample) < k:
            sample.append(row)
        else:
            j = rnd.randint(1, i)
            if j <= k:
                sample[j - 1] = row
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="Input manifest. Format: name=path.jsonl or path.jsonl (name derived from filename)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--concat", action="store_true", help="Concatenate inputs in order (default if no target_size)")
    ap.add_argument("--target_size", type=int, default=None, help="Target total samples; requires --ratios")
    ap.add_argument("--ratios", default=None, help="Comma list name=ratio; names must match --in names")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle within each source during sampling (reservoir)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--add_dataset_tag", action="store_true", help="Add dataset field with source name to each row")
    args = ap.parse_args()

    pairs = parse_in_args(args.inputs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.target_size and not args.ratios:
        raise SystemExit("--target_size requires --ratios")

    if not args.target_size:
        # Concatenate streaming
        n = 0
        with out_path.open("w", encoding="utf-8") as fo:
            for name, p in pairs:
                for row in iter_jsonl(p):
                    if args.add_dataset_tag:
                        row.setdefault("dataset", name)
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
        print(f"[OK] Concatenated -> {out_path} rows={n}")
        return

    # Sample to target size with ratios
    ratios = parse_ratios(args.ratios)
    missing = [nm for (nm, _p) in pairs if nm not in ratios]
    if missing:
        raise SystemExit(f"Ratios missing names: {missing}")
    target = int(args.target_size)
    plan = {nm: int(round(ratios[nm] * target)) for (nm, _p) in pairs}
    # Adjust rounding to hit exact target
    diff = target - sum(plan.values())
    if diff != 0:
        # Distribute diff to first items
        for nm, _p in pairs:
            if diff == 0:
                break
            plan[nm] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1

    rng = random.Random(args.seed)
    # Collect samples (reservoir per file)
    buckets: List[dict] = []
    for nm, p in pairs:
        k = max(0, plan[nm])
        if k == 0:
            continue
        rows = reservoir_sample(p, k, seed=(args.seed + hash(nm) % 997))
        if args.add_dataset_tag:
            for r in rows:
                r.setdefault("dataset", nm)
        buckets.extend(rows)

    if args.shuffle:
        rng.shuffle(buckets)

    with out_path.open("w", encoding="utf-8") as fo:
        for r in buckets[:target]:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Sampled -> {out_path} rows={min(len(buckets), target)} plan={plan}")


if __name__ == "__main__":
    main()

