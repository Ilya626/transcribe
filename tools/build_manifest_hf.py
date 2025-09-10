#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build JSONL manifests from Hugging Face datasets.

Each output line: {"audio_filepath": "<path>", "text": "<transcript>"}

Usage examples:
  # Common Voice 17 (RU) — все сплиты
  python tools/build_manifest_hf.py --preset cv17-ru --out data/cv17_ru.jsonl --drop_empty

  # RuLibriSpeech (путь 'bond005/rulibrispeech' или 'bond005___rulibrispeech')
  python tools/build_manifest_hf.py --preset rulibrispeech --out data/rulibri.jsonl --drop_empty

  # Произвольный датасет/конфиг/сплит
  python tools/build_manifest_hf.py \
      --name mozilla-foundation/common_voice_17_0 --config ru \
      --split train+validation+test --out data/cv17_ru.jsonl

Notes:
  - По умолчанию включен trust_remote_code=True (можно отключить флагом --no-trust-remote-code).
  - Для фильтров по длительности можно передать --min-sec / --max-sec (требует soundfile).
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Lazy imports (datasets может не стоять у некоторых окружений)
from datasets import load_dataset


# ----------------------------- Presets --------------------------------- #

def normalize_repo_id(repo_id: str) -> str:
    """
    Позволяет использовать строку вида 'user___dataset' как 'user/dataset'.
    Ничего не меняет, если уже есть '/'.
    """
    if "___" in repo_id and "/" not in repo_id:
        return repo_id.replace("___", "/")
    return repo_id


PRESETS: Dict[str, Dict[str, Any]] = {
    # Common Voice 17 RU (все сплиты)
    "cv17-ru": {
        "name": "mozilla-foundation/common_voice_17_0",
        "config": "ru",
        "split": "train+validation+test",
        "mapper": "common_voice17",
    },
    # RuLibriSpeech (комьюнити-репозиторий)
    "rulibrispeech": {
        "name": "bond005/rulibrispeech",  # допускаем и bond005___rulibrispeech (см. normalize_repo_id)
        "config": None,
        "split": "train+validation+test",  # если в репо нет всех трёх — укажи нужные вручную через --split
        "mapper": "generic_text_audio",
    },
    # Добавляй свои:
    # "my-preset": {"name": "...", "config": "...", "split": "...", "mapper": "generic_text_audio"},
}


# --------------------------- Mappers ----------------------------------- #

def _get_audio_path(ex: Dict[str, Any]) -> Optional[str]:
    """
    Универсальный способ вытащить путь к аудио.
    Приоритет: ex["audio"]["path"] -> ex["path"] -> ex["audio_filepath"]
    """
    audio = ex.get("audio")
    if isinstance(audio, dict):
        p = audio.get("path")
        if p:
            return p
    p = ex.get("path")
    if isinstance(p, str) and p:
        return p
    p = ex.get("audio_filepath")
    if isinstance(p, str) and p:
        return p
    return None


def _get_text(ex: Dict[str, Any]) -> str:
    """
    Универсальный способ вытащить текст.
    Частые поля: "sentence" (CV), "text", "normalized_text", "transcript".
    """
    for k in ("sentence", "text", "normalized_text", "normalized", "transcript", "transcription"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Попробуем что-то читаемое в одном из строковых полей
    for k, v in ex.items():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def map_common_voice17(ex: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Специальный маппер для Common Voice 17.
    """
    path = _get_audio_path(ex)
    text = ex.get("sentence") or _get_text(ex)
    if not path:
        return None
    return {"audio_filepath": path, "text": text or ""}


def map_generic_text_audio(ex: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Маппер "по умолчанию": ищет аудио и текст в типичных местах.
    Подходит для многих комьюнити-датасетов, включая RuLibriSpeech.
    """
    path = _get_audio_path(ex)
    text = _get_text(ex)
    if not path:
        return None
    return {"audio_filepath": path, "text": text or ""}


MAPPER_FUNCS = {
    "common_voice17": map_common_voice17,
    "generic_text_audio": map_generic_text_audio,
}


# ----------------------- Duration (optional) --------------------------- #

def compute_duration_seconds(path: str) -> Optional[float]:
    """
    По желанию: оценка длительности через soundfile (без декодирования в numpy).
    Возвращает None, если не удалось.
    """
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
        # иногда soundfile не даёт frames — попытаемся прочесть заголовок
        return getattr(info, "duration", None)
    except Exception:
        return None


# ------------------------------ Main ----------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # Способ 1: пресет
    ap.add_argument("--preset", type=str, default=None, help=f"One of: {', '.join(PRESETS.keys())}")

    # Способ 2: ручное задание
    ap.add_argument("--name", type=str, default=None, help="HF dataset repo id, e.g. mozilla-foundation/common_voice_17_0")
    ap.add_argument("--config", type=str, default=None, help="HF dataset config (language, etc.)")
    ap.add_argument("--split", type=str, default="train", help="split spec, e.g. 'train' or 'train+validation+test'")

    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")

    ap.add_argument("--drop_empty", action="store_true", help="Skip rows with empty text or missing audio")
    ap.add_argument("--min-sec", type=float, default=None, help="Keep only samples with duration >= min-sec")
    ap.add_argument("--max-sec", type=float, default=None, help="Keep only samples with duration <= max-sec")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows written")

    # trust_remote_code по умолчанию ВКЛ — чтобы CV17 и комьюнити-датасеты грузились без ошибки
    ap.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True,
                    help="Allow running dataset repo code (default: True)")
    ap.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false",
                    help="Disable running dataset repo code")

    ap.add_argument("--mapper", type=str, default=None,
                    help=f"Force mapper: {', '.join(MAPPER_FUNCS.keys())}. If omitted, chosen by preset or heuristics.")

    return ap


def choose_mapper(name: str, forced: Optional[str]) -> str:
    if forced:
        if forced not in MAPPER_FUNCS:
            raise ValueError(f"Unknown mapper '{forced}'. Available: {list(MAPPER_FUNCS.keys())}")
        return forced

    n = name.lower()
    if "common_voice" in n:
        return "common_voice17"
    # по умолчанию — универсальный
    return "generic_text_audio"


def main():
    args = build_parser().parse_args()

    if args.preset:
        if args.preset not in PRESETS:
            raise SystemExit(f"Unknown preset '{args.preset}'. Available: {list(PRESETS.keys())}")
        p = PRESETS[args.preset]
        name = normalize_repo_id(p["name"])
        config = p.get("config")
        split = p.get("split") or args.split
        mapper_key = p.get("mapper")
    else:
        if not args.name:
            raise SystemExit("Either --preset or --name must be provided.")
        name = normalize_repo_id(args.name)
        config = args.config
        split = args.split
        mapper_key = None  # выберем ниже эвристикой

    if args.mapper:
        mapper_key = args.mapper
    if not mapper_key:
        mapper_key = choose_mapper(name, forced=None)

    mapper = MAPPER_FUNCS[mapper_key]

    # Подготовим выходную папку
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset: name='{name}', config='{config}', split='{split}', trust_remote_code={args.trust_remote_code}")
    ds = load_dataset(
        path=name,
        name=config,
        split=split,
        trust_remote_code=args.trust_remote_code,
    )

    total = len(ds)
    written = 0
    skipped_empty = 0
    skipped_dur = 0

    want_min = args.min_sec is not None
    want_max = args.max_sec is not None

    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            row = mapper(ex)
            if row is None:
                skipped_empty += 1
                continue

            # drop_empty: пустой текст или отсутствующий путь
            if args.drop_empty and (not row.get("audio_filepath") or not (row.get("text") or "").strip()):
                skipped_empty += 1
                continue

            # duration filters (optional)
            if (want_min or want_max) and row.get("audio_filepath"):
                dur = compute_duration_seconds(row["audio_filepath"])
                if dur is None:
                    # не удалось оценить — считаем, что подходит (или можно скипнуть, если нужно строго)
                    pass
                else:
                    if want_min and dur < float(args.min_sec):
                        skipped_dur += 1
                        continue
                    if want_max and dur > float(args.max_sec):
                        skipped_dur += 1
                        continue

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if args.max_rows is not None and written >= int(args.max_rows):
                print(f"[INFO] Reached --max-rows={args.max_rows}, stopping early.")
                break

            if (i + 1) % 500 == 0:
                print(f"[PROGRESS] {i + 1}/{total} processed, {written} written...")

    print(f"[DONE] Processed: {total}, written: {written}, skipped_empty: {skipped_empty}, skipped_dur: {skipped_dur}")
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()
