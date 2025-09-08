"""Evaluate ASR predictions against references using lexical and semantic metrics.

The script reports word, character and sentence error rates (WER, CER, SER),
breaks down lexical errors by substitution/insertion/deletion counts and also
computes semantic similarity via Sentence-Transformers.
"""
import argparse
import json
import os
import string
from pathlib import Path

from jiwer import cer, process_words
from sentence_transformers import SentenceTransformer, util


TRANSFORMER_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def normalize(text: str) -> str:
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def _ensure_local_caches():
    """Route HF/Torch caches to project-local folders if not already set.

    This ensures model downloads land under the repository directory and
    avoid user profile disks. Safe to call multiple times.
    """
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
        # create directories best-effort
        for d in [hf, os.environ["HF_HUB_CACHE"], os.environ["TORCH_HOME"], tmp]:
            Path(d).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def load_json_or_jsonl(path: Path) -> dict:
    """Load references as a mapping path->text from JSON or JSONL.

    - JSON: expects {"path": "text", ...}
    - JSONL: expects objects with keys {"audio_filepath", "text"}
    """
    if path.suffix.lower() == ".jsonl":
        refs: dict[str, str] = {}
        # support UTF-8 with BOM
        with open(path, "r", encoding="utf-8-sig") as f:
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
        # if it's a list of objects from a JSON (not JSONL), map it too
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=str, help="Path to reference transcripts JSON")
    parser.add_argument("predictions", type=str, help="Path to predictions JSON")
    parser.add_argument(
        "--model", default=TRANSFORMER_MODEL, help="Sentence-transformer model for semantic similarity"
    )
    args = parser.parse_args()

    _ensure_local_caches()
    refs = load_json_or_jsonl(Path(args.references))
    with open(Path(args.predictions), "r", encoding="utf-8") as f:
        preds = json.load(f)

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
