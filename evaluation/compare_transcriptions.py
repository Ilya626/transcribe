"""Compare multiple ASR prediction sets against references.

The script outputs WER, CER, SER, substitution/insertion/deletion counts and
semantic similarity for each predictions file.
"""
import argparse
import json
import os
import string
from pathlib import Path

from jiwer import cer, process_words
from sentence_transformers import SentenceTransformer, util


def normalize(text: str) -> str:
    """Lower-case and strip punctuation."""
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def load_json_or_jsonl(path: Path) -> dict:
    """Load mapping path->text from JSON or JSONL with audio_filepath/text."""
    if path.suffix.lower() == ".jsonl":
        refs: dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                k = obj.get("audio_filepath") or obj.get("audio")
                v = obj.get("text", "")
                if k is not None:
                    refs[str(k)] = v
        return refs
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            refs: dict[str, str] = {}
            for obj in data:
                if isinstance(obj, dict):
                    k = obj.get("audio_filepath") or obj.get("audio")
                    v = obj.get("text", "")
                    if k is not None:
                        refs[str(k)] = v
            if refs:
                return refs
        return data


def evaluate(
    refs: dict, preds: dict, model: SentenceTransformer
) -> tuple[float, float, float, float, int, int, int]:
    """Return WER, CER, SER, semantic similarity and error counts."""
    refs_norm, preds_norm = [], []
    for k, ref in refs.items():
        if k in preds:
            refs_norm.append(normalize(ref))
            preds_norm.append(normalize(preds[k]))
    if not refs_norm:
        raise ValueError("No matching keys between references and predictions")

    word_info = process_words(refs_norm, preds_norm)
    w_error = word_info.wer
    c_error = cer(refs_norm, preds_norm)
    ser = sum(r != p for r, p in zip(refs_norm, preds_norm)) / len(refs_norm)
    ref_emb = model.encode(refs_norm, convert_to_tensor=True)
    pred_emb = model.encode(preds_norm, convert_to_tensor=True)
    semantic = float(util.cos_sim(ref_emb, pred_emb).diagonal().mean())
    return (
        w_error,
        c_error,
        ser,
        semantic,
        word_info.substitutions,
        word_info.insertions,
        word_info.deletions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=str, help="Path to reference transcripts JSON")
    parser.add_argument("predictions", nargs="+", help="Prediction JSON files to compare")
    parser.add_argument(
        "--model",
        default="paraphrase-multilingual-mpnet-base-v2",
        help="Sentence-transformer model for semantic similarity",
    )
    args = parser.parse_args()

    _ensure_local_caches()
    refs = load_json_or_jsonl(Path(args.references))
def _ensure_local_caches():
    """Ensure HF/Torch caches live under the repo directory."""
    try:
        repo_root = Path(__file__).resolve().parents[1]
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
    except Exception:
        pass

    st_model = SentenceTransformer(args.model)

    for pred_path in args.predictions:
        with open(Path(pred_path), "r", encoding="utf-8") as f:
            preds = json.load(f)
        (
            wer_score,
            cer_score,
            ser_score,
            semantic,
            subs,
            ins,
            dels,
        ) = evaluate(refs, preds, st_model)
        label = Path(pred_path).stem
        print(f"\nResults for {label}:")
        print(f"WER: {wer_score:.4f}")
        print(f"CER: {cer_score:.4f}")
        print(f"SER: {ser_score:.4f}")
        print(
            f"Substitutions: {subs} | Insertions: {ins} | Deletions: {dels}"
        )
        print(f"Semantic similarity: {semantic:.4f}")


if __name__ == "__main__":
    main()
