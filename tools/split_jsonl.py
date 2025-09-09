#!/usr/bin/env python
"""
Split a JSONL manifest into train/val with optional stratification by a key.

Usage:
  python transcribe/tools/split_jsonl.py \
    --in data/ru_megamix_hard25_easy15_ns3.jsonl \
    --train data/train_megamix.jsonl --val data/val_megamix.jsonl \
    --val_ratio 0.1 --group_key dataset --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--group_key", default=None, help="Optional key to stratify (e.g., dataset)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.inp))
    rng = random.Random(int(args.seed))

    if not args.group_key:
        rng.shuffle(rows)
        k = int(round(len(rows) * float(args.val_ratio)))
        val = rows[:k]
        train = rows[k:]
    else:
        # Stratify by group_key
        by_g: Dict[str, List[dict]] = {}
        for r in rows:
            g = str(r.get(args.group_key, ""))
            by_g.setdefault(g, []).append(r)
        train: List[dict] = []
        val: List[dict] = []
        for g, lst in by_g.items():
            rng.shuffle(lst)
            k = int(round(len(lst) * float(args.val_ratio)))
            val.extend(lst[:k])
            train.extend(lst[k:])

    trn = write_jsonl(Path(args.train), train)
    vl = write_jsonl(Path(args.val), val)
    print(f"[OK] Wrote train={trn} -> {args.train}; val={vl} -> {args.val}")


if __name__ == "__main__":
    main()

