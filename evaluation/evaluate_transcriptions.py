"""Evaluate ASR predictions against references using lexical and semantic metrics.

The script reports word, character and sentence error rates (WER, CER, SER),
breaks down lexical errors by substitution/insertion/deletion counts and also
computes semantic similarity via Sentence-Transformers.
"""
import argparse
import json
import string
from pathlib import Path

from jiwer import cer, process_words
from sentence_transformers import SentenceTransformer, util


TRANSFORMER_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def normalize(text: str) -> str:
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=str, help="Path to reference transcripts JSON")
    parser.add_argument("predictions", type=str, help="Path to predictions JSON")
    parser.add_argument(
        "--model", default=TRANSFORMER_MODEL, help="Sentence-transformer model for semantic similarity"
    )
    args = parser.parse_args()

    refs = load_json(Path(args.references))
    preds = load_json(Path(args.predictions))

    refs_norm = []
    preds_norm = []
    ids = []
    for k, ref in refs.items():
        if k in preds:
            ids.append(k)
            refs_norm.append(normalize(ref))
            preds_norm.append(normalize(preds[k]))

    if not ids:
        raise ValueError("No matching keys between references and predictions")

    # Lexical metrics
    word_info = process_words(refs_norm, preds_norm)
    w_error = word_info.wer
    c_error = cer(refs_norm, preds_norm)
    ser = sum(r != p for r, p in zip(refs_norm, preds_norm)) / len(refs_norm)
    print(
        f"WER: {w_error:.4f}\nCER: {c_error:.4f}\nSER: {ser:.4f}"
    )
    print(
        f"Substitutions: {word_info.substitutions} | "
        f"Insertions: {word_info.insertions} | Deletions: {word_info.deletions}"
    )

    model = SentenceTransformer(args.model)
    ref_emb = model.encode(refs_norm, convert_to_tensor=True)
    pred_emb = model.encode(preds_norm, convert_to_tensor=True)
    cosine_scores = util.cos_sim(ref_emb, pred_emb).diagonal()
    semantic = float(cosine_scores.mean())
    print(f"Semantic similarity: {semantic:.4f}")


if __name__ == "__main__":
    main()
