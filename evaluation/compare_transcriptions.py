"""Compare multiple ASR prediction sets against references using lexical and semantic metrics."""
import argparse
import json
import string
from pathlib import Path

from jiwer import wer, cer
from sentence_transformers import SentenceTransformer, util


def normalize(text: str) -> str:
    """Lower-case and strip punctuation."""
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(refs: dict, preds: dict, model: SentenceTransformer) -> tuple[float, float, float]:
    """Return WER, CER and semantic similarity for matching keys."""
    refs_norm, preds_norm = [], []
    for k, ref in refs.items():
        if k in preds:
            refs_norm.append(normalize(ref))
            preds_norm.append(normalize(preds[k]))
    if not refs_norm:
        raise ValueError("No matching keys between references and predictions")

    w_error = wer(refs_norm, preds_norm)
    c_error = cer(refs_norm, preds_norm)
    ref_emb = model.encode(refs_norm, convert_to_tensor=True)
    pred_emb = model.encode(preds_norm, convert_to_tensor=True)
    semantic = float(util.cos_sim(ref_emb, pred_emb).diagonal().mean())
    return w_error, c_error, semantic


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

    refs = load_json(Path(args.references))
    st_model = SentenceTransformer(args.model)

    for pred_path in args.predictions:
        preds = load_json(Path(pred_path))
        wer_score, cer_score, semantic = evaluate(refs, preds, st_model)
        label = Path(pred_path).stem
        print(f"\nResults for {label}:")
        print(f"WER: {wer_score:.4f}")
        print(f"CER: {cer_score:.4f}")
        print(f"Semantic similarity: {semantic:.4f}")


if __name__ == "__main__":
    main()
