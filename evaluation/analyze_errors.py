import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transcribe.evaluation.advanced.align import (
    NormConfig,
    align_tokens,
    compute_metrics,
    extract_confusions,
    normalize_text,
    tokenize,
)
from transcribe.evaluation.advanced.taxonomy import categorize_ops, classify_utterance
from transcribe.evaluation.advanced.ner import extract_entities
from transcribe.evaluation.advanced.semantics import cosine_similarity, cosine_sims_for_pairs
from transcribe.evaluation.advanced.report import (
    aggregate_model_metrics,
    top_confusions,
    write_csv,
    write_html_summary,
    write_json,
)


@dataclass
class Example:
    utt_id: str
    ref: str


def load_references(path: Path) -> Dict[str, Example]:
    if path.suffix.lower() == ".jsonl":
        refs = {}
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                uid = o.get("audio_filepath") or o.get("id") or o.get("utt_id")
                if not uid:
                    continue
                refs[uid] = Example(utt_id=uid, ref=str(o.get("text", "")))
        return refs
    else:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        refs = {k: Example(utt_id=k, ref=str(v)) for k, v in obj.items()}
        return refs


def load_predictions(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): str(v) for k, v in obj.items()}


def parse_pred_arg(arg: str) -> Tuple[str, Path]:
    if "=" in arg:
        mid, p = arg.split("=", 1)
        return mid.strip(), Path(p)
    p = Path(arg)
    mid = p.stem
    return mid, p


def main():
    ap = argparse.ArgumentParser(description="Advanced ASR error analysis")
    ap.add_argument("reference", type=Path, help="Reference file: JSON or JSONL")
    ap.add_argument("--pred", action="append", default=[], help="Prediction as model_id=path.json (can repeat)")
    ap.add_argument("--outdir", type=Path, default=Path("transcribe/preds/analysis"))
    ap.add_argument("--no_semantic", action="store_true")
    ap.add_argument("--no_ner", action="store_true")
    ap.add_argument("--st_model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--st_device", default="cpu", choices=["cpu", "cuda"], help="Device for sentence-transformers model")
    ap.add_argument("--st_bs", type=int, default=256, help="Batch size for sentence-transformers encoding")
    ap.add_argument("--sample", type=int, default=0, help="Sample N utterances (intersection across preds)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    refs = load_references(args.reference)
    pred_specs = [parse_pred_arg(s) for s in args.pred]
    if not pred_specs:
        raise SystemExit("Provide at least one --pred model_id=path.json")

    cfg = NormConfig()
    per_row: List[Dict] = []
    taxonomy_totals: Dict[str, int] = {}

    # Determine intersection keys if sampling requested
    ref_keys = set(refs.keys())
    pred_maps: List[Tuple[str, Dict[str, str]]] = []
    for model_id, pred_path in pred_specs:
        pred_maps.append((model_id, load_predictions(pred_path)))
    if args.sample and args.sample > 0:
        inter = ref_keys.copy()
        for _, pm in pred_maps:
            inter &= set(pm.keys())
        keys = list(inter)
        if not keys:
            raise SystemExit("No intersection between reference and predictions keys")
        import random
        random.seed(args.seed)
        random.shuffle(keys)
        keys = keys[: args.sample]
        refs = {k: refs[k] for k in keys}

    # First pass (no semantic): compute alignment/metrics, store normalized texts for later semantic batch
    for model_id, preds in pred_maps:
        for uid, ex in refs.items():
            ref_raw = ex.ref
            hyp_raw = preds.get(uid, "")
            ref = normalize_text(ref_raw, cfg)
            hyp = normalize_text(hyp_raw, cfg)
            ops = align_tokens(tokenize(ref), tokenize(hyp))
            metrics = compute_metrics(ref, hyp)
            cats = categorize_ops(ops)
            for k, v in cats.items():
                taxonomy_totals[k] = taxonomy_totals.get(k, 0) + int(v)
            row = {
                "utt_id": uid,
                "model_id": model_id,
                "ref_norm": ref,
                "hyp_norm": hyp,
                "wer": round(metrics["wer"], 6),
                "cer": round(metrics["cer"], 6),
                "ser": round(metrics["ser"], 6),
                "sim": None,
                "class": None,
                "ops_sub": cats.get("sub", 0),
                "ops_ins": cats.get("ins", 0),
                "ops_del": cats.get("del", 0),
                "ops_correct": cats.get("correct", 0),
                "cat_number": cats.get("cat_number", 0),
                "cat_abbr": cats.get("cat_abbr", 0),
                "cat_latin": cats.get("cat_latin", 0),
                "cat_word": cats.get("cat_word", 0),
                "confusions": extract_confusions(ops),
            }
            if not args.no_ner:
                ents = extract_entities(hyp_raw)
                row["ner_count"] = len(ents)
            per_row.append(row)

    # Second pass: batch semantic similarities
    if not args.no_semantic and per_row:
        pairs = [(r["ref_norm"], r["hyp_norm"]) for r in per_row]
        print(f"[SEM] Computing semantic similarity for {len(pairs)} pairs on {args.st_device} ...")
        sims = cosine_sims_for_pairs(pairs, model_name=args.st_model, device=args.st_device, batch_size=args.st_bs)
        for r, s in zip(per_row, sims):
            r["sim"] = None if s is None else round(float(s), 6)

    # Third pass: classify now that sim is known
    for r in per_row:
        r["class"] = classify_utterance({"wer": r["wer"], "cer": r["cer"], "ser": r["ser"]}, r.get("sim"))

    # Экспорт построчных результатов; конфузии как строка "a->b;..."
    per_rows_flat: List[Dict] = []
    for r in per_row:
        rr = {k: v for k, v in r.items() if k != "confusions"}
        conf_list = r.get("confusions") or []
        rr["confusions"] = ";".join([f"{a}->{b}" for a, b in conf_list])
        per_rows_flat.append(rr)
    write_csv(outdir / "per_utterance.csv", per_rows_flat)

    model_aggs = aggregate_model_metrics(per_row)
    write_csv(outdir / "models.csv", model_aggs)

    conf = top_confusions(per_row, top_k=100)
    write_csv(outdir / "top_confusions.csv", [{"ref": a, "hyp": b, "count": n} for a, b, n in conf])

    total_ops = sum(v for k, v in taxonomy_totals.items() if k in ("sub", "ins", "del", "correct")) or 1
    taxonomy_share = {k: v / total_ops for k, v in taxonomy_totals.items() if k in ("sub", "ins", "del")}
    write_json(outdir / "taxonomy.json", taxonomy_share)

    # Disagreement per utterance
    by_utt: Dict[str, List[Dict]] = {}
    for r in per_row:
        by_utt.setdefault(r["utt_id"], []).append(r)
    dis_rows: List[Dict] = []
    for uid, lst in by_utt.items():
        wers = [float(r.get("wer", 0.0)) for r in lst]
        mean = sum(wers) / max(1, len(wers))
        var = sum((w - mean) ** 2 for w in wers) / max(1, len(wers))
        std = var ** 0.5
        hyps = set()
        for r in lst:
            hyps.add((r["model_id"], r.get("wer")))
        dis_rows.append({"utt_id": uid, "std_wer": round(std, 6), "models": len(lst), "distinct_scores": len(hyps)})
    write_csv(outdir / "disagreement.csv", dis_rows)

    # Baseline pattern and relative score per model
    def _vec(x: Dict) -> List[float]:
        return [float(x.get("wer", 0.0)), float(x.get("cer", 0.0)), float(x.get("ser", 0.0)), float(x.get("sim", 0.0) or 0.0)]

    def _cos(a: List[float], b: List[float]) -> float:
        import math
        na = math.sqrt(sum(v * v for v in a)) or 1.0
        nb = math.sqrt(sum(v * v for v in b)) or 1.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    if model_aggs:
        base = [0.0, 0.0, 0.0, 0.0]
        for r in model_aggs:
            v = _vec(r)
            base = [a + b for a, b in zip(base, v)]
        base = [a / len(model_aggs) for a in base]
        for r in model_aggs:
            v = _vec(r)
            r["baseline_cosine"] = round(_cos(v, base), 6)
            r["delta_wer"] = round(r.get("wer", 0.0) - base[0], 6)
            r["delta_cer"] = round(r.get("cer", 0.0) - base[1], 6)
            r["delta_ser"] = round(r.get("ser", 0.0) - base[2], 6)
    write_csv(outdir / "models_with_baseline.csv", model_aggs)

    write_html_summary(outdir / "summary.html", model_aggs, conf, taxonomy_share)

    write_json(outdir / "summary.json", {
        "models": model_aggs,
        "taxonomy_share": taxonomy_share,
        "top_confusions": conf[:20],
        "disagreement_file": str((outdir / "disagreement.csv").as_posix()),
    })

    # release semantic model explicitly (if loaded)
    try:
        from transcribe.evaluation.advanced.semantics import release_semantic_models

        release_semantic_models()
    except Exception:
        pass


if __name__ == "__main__":
    main()
