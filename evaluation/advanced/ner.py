from typing import List, Tuple


def _try_natasha(text: str) -> List[Tuple[str, str]]:
    try:
        from natasha import Doc, NewsNERTagger, NewsEmbedding, Segmenter
    except Exception:
        return []
    emb = NewsEmbedding()
    ner = NewsNERTagger(emb)
    seg = Segmenter()
    doc = Doc(text if isinstance(text, str) else str(text))
    doc.segment(seg)
    # Use Natasha Doc API to tag NER safely
    doc.tag_ner(ner)
    ents = []
    for span in doc.spans or []:
        try:
            ents.append((span.text, span.type))
        except Exception:
            pass
    return ents


def _heuristic_ru_caps(text: str) -> List[Tuple[str, str]]:
    ents = []
    prev_end = True
    for token in text.split():
        if token[:1].isupper() and not token.isupper():
            ents.append((token, "PER?"))
        prev_end = token.endswith(".")
    return ents


def extract_entities(text: str) -> List[Tuple[str, str]]:
    ents = _try_natasha(text)
    if ents:
        return ents
    return _heuristic_ru_caps(text)
