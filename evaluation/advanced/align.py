from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import jiwer


@dataclass
class NormConfig:
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_multiple_spaces: bool = True
    strip: bool = True


def normalize_text(text: str, cfg: Optional[NormConfig] = None) -> str:
    if cfg is None:
        cfg = NormConfig()
    # Unify ё/Ё to е/Е
    if text is None:
        text = ""
    text = text.replace("ё", "е").replace("Ё", "Е")
    transforms = []
    if cfg.lowercase:
        transforms.append(jiwer.ToLowerCase())
    if cfg.remove_punctuation:
        transforms.append(jiwer.RemovePunctuation())
    if cfg.remove_multiple_spaces:
        transforms.append(jiwer.RemoveMultipleSpaces())
    if cfg.strip:
        transforms.append(jiwer.Strip())
    transform = jiwer.Compose(transforms) if transforms else (lambda x: x)
    return transform(text)


def tokenize(text: str) -> List[str]:
    return [t for t in text.split() if t]


@dataclass
class Op:
    op: str  # one of: correct, sub, ins, del
    ref: Optional[str]
    hyp: Optional[str]


def _argmin3(a: Tuple[int, int, int]) -> int:
    # return index of min
    m = min(a)
    return a.index(m)


def align_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> List[Op]:
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]  # backtrace: 0=sub/ok, 1=del, 2=ins
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = 2
    for i in range(1, n + 1):
        rt = ref_tokens[i - 1]
        for j in range(1, m + 1):
            ht = hyp_tokens[j - 1]
            cost_sub = dp[i - 1][j - 1] + (0 if rt == ht else 1)
            cost_del = dp[i - 1][j] + 1
            cost_ins = dp[i][j - 1] + 1
            k = _argmin3((cost_sub, cost_del, cost_ins))
            dp[i][j] = (cost_sub, cost_del, cost_ins)[k]
            bt[i][j] = k
    ops: List[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        k = bt[i][j]
        if k == 0:
            rt = ref_tokens[i - 1] if i > 0 else None
            ht = hyp_tokens[j - 1] if j > 0 else None
            ops.append(Op(op=("correct" if rt == ht else "sub"), ref=rt, hyp=ht))
            i -= 1
            j -= 1
        elif k == 1:
            rt = ref_tokens[i - 1] if i > 0 else None
            ops.append(Op(op="del", ref=rt, hyp=None))
            i -= 1
        elif k == 2:
            ht = hyp_tokens[j - 1] if j > 0 else None
            ops.append(Op(op="ins", ref=None, hyp=ht))
            j -= 1
        else:
            break
    ops.reverse()
    return ops


def _edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def compute_metrics(ref: str, hyp: str) -> Dict[str, float]:
    ref_tokens = tokenize(ref)
    hyp_tokens = tokenize(hyp)
    ops = align_tokens(ref_tokens, hyp_tokens)
    subs = sum(1 for o in ops if o.op == "sub")
    ins = sum(1 for o in ops if o.op == "ins")
    dels = sum(1 for o in ops if o.op == "del")
    denom = max(1, len(ref_tokens))
    wer = float(subs + ins + dels) / denom
    ser = 0.0 if ref.strip() == hyp.strip() else 1.0
    # CER: нормированный символный edit distance
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    dist = _edit_distance(ref_chars, hyp_chars)
    cer = float(dist) / max(1, len(ref_chars))
    return {"wer": wer, "cer": cer, "ser": float(ser)}


def extract_confusions(ops: List[Op]) -> List[Tuple[str, str]]:
    conf = []
    for o in ops:
        if o.op == "sub" and o.ref is not None and o.hyp is not None:
            conf.append((o.ref, o.hyp))
    return conf
