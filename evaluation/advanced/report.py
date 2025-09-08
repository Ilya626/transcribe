import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def aggregate_model_metrics(rows: List[Dict]) -> List[Dict]:
    agg = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    for r in rows:
        mid = r["model_id"]
        for k in ("wer", "cer", "ser", "sim"):
            v = r.get(k)
            if v is not None:
                agg[mid][k] += float(v)
        cnt[mid] += 1
    out = []
    for mid, sums in agg.items():
        n = max(1, cnt[mid])
        out.append({
            "model_id": mid,
            "wer": sums.get("wer", 0.0) / n,
            "cer": sums.get("cer", 0.0) / n,
            "ser": sums.get("ser", 0.0) / n,
            "sim": sums.get("sim", 0.0) / n,
            "count": n,
        })
    out.sort(key=lambda x: x.get("wer", 0.0))
    return out


def top_confusions(rows: List[Dict], top_k: int = 50) -> List[Tuple[str, str, int]]:
    c = Counter()
    for r in rows:
        for a, b in r.get("confusions", []) or []:
            c[(a, b)] += 1
    items = [(*k, v) for k, v in c.most_common(top_k)]
    return items


def write_html_summary(path: Path, model_aggs: List[Dict], top_conf: List[Tuple[str, str, int]], taxonomy: Dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    def table(rows: List[Dict]) -> str:
        if not rows:
            return "<p>No data</p>"
        cols = list(rows[0].keys())
        th = "".join(f"<th>{c}</th>" for c in cols)
        trs = []
        for r in rows:
            tds = "".join(f"<td>{r.get(c, '')}</td>" for c in cols)
            trs.append(f"<tr>{tds}</tr>")
        return f"<table border=1><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"

    conf_rows = [{"ref": a, "hyp": b, "count": n} for a, b, n in top_conf]
    tax_rows = [{"category": k, "share": round(v, 4)} for k, v in taxonomy.items()]
    html = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>ASR Evaluation Summary</title></head>
<body>
<h1>Summary</h1>
<h2>Model Metrics</h2>
{table(model_aggs)}
<h2>Error Taxonomy Shares</h2>
{table(tax_rows)}
<h2>Top Confusions</h2>
{table(conf_rows)}
</body></html>
"""
    path.write_text(html, encoding="utf-8")

