from typing import Optional, List, Tuple, Dict


class _STModel:
    def __init__(self, name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)

    def embed(self, s: str):
        return self.model.encode([s], normalize_embeddings=True)[0]

    def embed_many(self, texts: List[str], batch_size: int = 256, show_progress_bar: bool = True):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=show_progress_bar)


def _cosine(a, b) -> float:
    import numpy as np
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float((a * b).sum())


def cosine_similarity(ref: str, hyp: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu", _cache: dict = {}) -> Optional[float]:
    try:
        key = (model_name, device)
        mdl = _cache.get(key)
        if mdl is None:
            mdl = _STModel(model_name, device=device)
            _cache[key] = mdl
        a = mdl.embed(ref)
        b = mdl.embed(hyp)
        return _cosine(a, b)
    except Exception:
        return None


def cosine_sims_for_pairs(
    pairs: List[Tuple[str, str]],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    device: str = "cpu",
    batch_size: int = 256,
    _cache: dict = {},
) -> List[Optional[float]]:
    try:
        key = (model_name, device)
        mdl = _cache.get(key)
        if mdl is None:
            mdl = _STModel(model_name, device=device)
            _cache[key] = mdl
        # Deduplicate texts
        uniq: Dict[str, int] = {}
        texts: List[str] = []
        for a, b in pairs:
            for t in (a, b):
                if t not in uniq:
                    uniq[t] = len(texts)
                    texts.append(t)
        import numpy as np

        embs = mdl.embed_many(texts, batch_size=batch_size, show_progress_bar=True)
        embs = np.asarray(embs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms
        out: List[float] = []
        for a, b in pairs:
            ia = uniq[a]
            ib = uniq[b]
            out.append(float((embs[ia] * embs[ib]).sum()))
        return out
    except Exception:
        return [None for _ in pairs]


def bert_score_for_pairs(
    pairs: List[Tuple[str, str]],
    model_type: str = "bert-base-multilingual-cased",
    device: str = "cpu",
    batch_size: int = 64,
    lang: str = "en",
) -> List[Optional[float]]:
    try:
        from bert_score import score as bert_score

        refs = [a for a, _ in pairs]
        hyps = [b for _, b in pairs]
        _, _, f1 = bert_score(hyps, refs, model_type=model_type, device=device, batch_size=batch_size, lang=lang)
        return [float(v) for v in f1.tolist()]
    except Exception:
        return [None for _ in pairs]


def release_semantic_models(_cache: dict = {}):
    try:
        keys = list(_cache.keys())
        for k in keys:
            try:
                del _cache[k]
            except Exception:
                pass
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass
