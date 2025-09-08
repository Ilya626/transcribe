import re
from typing import Dict, List, Tuple

from .align import Op


def is_number(token: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+[\d\s,.]*", token))


def is_abbrev(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-zА-ЯЁЇІЄҐ]+\.?", token)) and token.isupper()


def token_category(token: str) -> str:
    if token is None or token == "":
        return "empty"
    if is_number(token):
        return "number"
    if is_abbrev(token):
        return "abbr"
    if re.search(r"[A-Za-z]", token):
        return "latin"
    return "word"


def classify_utterance(metrics: Dict[str, float], sim: float = None, thr_wer_hi: float = 0.4, thr_wer_lo: float = 0.15, thr_sim_lo: float = 0.6) -> str:
    wer = metrics.get("wer", 0.0)
    if sim is None:
        if wer >= thr_wer_hi:
            return "full_misunderstanding"
        if wer >= thr_wer_lo:
            return "partial"
        return "local"
    if wer >= thr_wer_hi and sim <= thr_sim_lo:
        return "full_misunderstanding"
    if wer >= thr_wer_lo:
        return "partial"
    return "local"


def categorize_ops(ops: List[Op]) -> Dict[str, int]:
    counts = {"sub": 0, "ins": 0, "del": 0, "correct": 0}
    cats = {"number": 0, "abbr": 0, "latin": 0, "word": 0}
    for o in ops:
        counts[o.op] = counts.get(o.op, 0) + 1
        t = o.ref if o.ref is not None else (o.hyp or "")
        cats[token_category(t)] = cats.get(token_category(t), 0) + 1
    return {**counts, **{f"cat_{k}": v for k, v in cats.items()}}

