#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build JSONL manifests from Hugging Face datasets.

Каждая строка вывода: {"audio_filepath": "<path>", "text": "<transcript>"}

Примеры:
  # Common Voice 17 (RU) — все сплиты
  python tools/build_manifest_hf.py --preset cv17-ru --out data/cv17_ru.jsonl --drop_empty

  # RuLibriSpeech (алиас: ruls, mls-ru)
  python tools/build_manifest_hf.py --preset ruls --out data/ruls.jsonl --drop_empty

  # FLEURS (ru_ru)
  python tools/build_manifest_hf.py --preset fleurs-ru --out data/fleurs_ru.jsonl --drop_empty

  # GOLOS (crowd/farfield)
  python tools/build_manifest_hf.py --preset golos-crowd --out data/golos_crowd.jsonl --drop_empty
  python tools/build_manifest_hf.py --preset golos-farfield --out data/golos_farfield.jsonl --drop_empty

  # Podlodka
  python tools/build_manifest_hf.py --preset podlodka --out data/podlodka.jsonl --drop_empty

  # Произвольный датасет/конфиг/сплит с кастомными колонками
  python tools/build_manifest_hf.py \
      --dataset UniDataPro/russian-speech-recognition-dataset \
      --split train+validation+test --audio_col audio --text_col transcript \
      --out data/telephony.jsonl --drop_empty

Notes:
  - По умолчанию trust_remote_code=True (можно отключить --no-trust-remote-code).
  - Для фильтров по длительности можно передать --min-sec / --max-sec (нужен soundfile).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset  # убедитесь, что в requirements есть: datasets>=2.19.0
# soundfile нужен только если используете фильтры по длительности


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

    # Russian LibriSpeech (RuLS). В MLS нет русского — используем версию bond005.
    "ruls": {
        "name": "bond005/rulibrispeech",
        "config": None,
        "split": "train+validation+test",
        "mapper": "generic_text_audio",
    },
    "rulibrispeech": {  # синоним
        "name": "bond005/rulibrispeech",
        "config": None,
        "split": "train+validation+test",
        "mapper": "generic_text_audio",
    },
    "mls-ru": {  # алиас на RuLS, т.к. в facebook/multilingual_librispeech нет 'ru'
        "name": "bond005/rulibrispeech",
        "config": None,
        "split": "train+validation+test",
        "mapper": "generic_text_audio",
    },

    # FLEURS RU
    "fleurs-ru": {
        "name": "google/fleurs",
        "config": "ru_ru",
        "split": "train+validation+test",
        "mapper": "generic_text_audio",  # у FLEURS поля path/audio/transcription
    },

    # GOLOS (адаптированные сабсеты на HF)
    "golos-crowd": {
        "name": "bond005/sberdevices_golos_10h_crowd",
        "config": None,
        "split": "train+test",
        "mapper": "generic_text_audio",
    },
    "golos-farfield": {
        "name": "bond005/sberdevices_golos_100h_farfield",
        "config": None,
        "split": "train+test",
        "mapper": "generic_text_audio",
    },

    # Podlodka
    "podlodka": {
        "name": "bond005/podlodka_speech",
        "config": None,
        "split": "train+validation+test",
        "mapper": "generic_text_audio",
    },
}


# --------------------------- Mappers ----------------------------------- #

def _extract_audio_path(value: Any) -> Optional[str]:
    """
    Приводим разные варианты к строке пути:
      - dict (Audio feature): берем value["path"]
      - str: возвращаем как есть
    """
    if isinstance(value, dict):
        p = value.get("path")
        return p if isinstance(p, str) and p else None
    if isinstance(value, str) and value:
        return value
    return None


def _get_audio_path(ex: Dict[str, Any]) -> Optional[str]:
    """
    Универсальный способ вытащить путь к аудио.
    Приоритет: ex["audio"]["path"] -> ex["path"] -> ex["file"] -> ex["audio_filepath"]
    """
    for key in ("audio", "path", "file", "audio_filepath"):
        if key in ex:
            p = _extract_audio_path(ex[key])
            if p:
                return p
    return None


def _get_text(ex: Dict[str, Any]) -> str:
    """
    Универсальный способ вытащить текст.
    Частые поля: "sentence" (CV), "text", "normalized_text", "transcript(ion)".
    """
    for k in ("sentence", "text", "normalized_text", "normalized", "transcript", "transcription", "raw_transcription"):
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
    Подходит для многих датасетов (RuLS, FLEURS, GOLOS-сабсеты, Podlodka и т.п.).
    """
    path = _get_audio_path(ex)
    text = _get_text(ex)
    if not path:
        return None
    return {"audio_filepath": path, "text": text or ""}


def make_custom_mapper(audio_col: str, text_col: str):
    """
    Маппер с явными названиями колонок (для --audio_col/--text_col).
    """
    a_col = audio_col.strip()
    t_col = text_col.strip()

    def _mapper(ex: Dict[str, Any]) -> Optional[Dict[str, str]]:
        if a_col not in ex or t_col not in ex:
            return None
        path = _extract_audio_path(ex[a_col])
        if not path:
            return None
        text_val = ex[t_col]
        if isinstance(text_val, str):
            txt = text_val.strip()
        else:
            txt = str(text_val) if text_val is not None else ""
        return {"audio_filepath": path, "text": txt}

    return _mapper


MAPPER_FUNCS = {
    "common_voice17": map_common_voice17,
    "generic_text_audio": map_generic_text_audio,
}


# ----------------------- Duration (optional) --------------------------- #

def compute_duration_seconds(path: str) -> Optional[float]:
    """
    Оценка длительности через soundfile без полного декодирования.
    Возвращает None, если не удалось (или soundfile не установлен).
    """
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(path)
        if getattr(info, "frames", 0) and getattr(info, "samplerate", 0):
            return float(info.frames) / float(info.samplerate)
        return getattr(info, "duration", None)
    except Exception:
        return None


# ------------------------------ Main ----------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # Вариант 1: пресет (один) или список пресетов
    ap.add_argument("--preset", type=str, default=None, help=f"One of: {', '.join(sorted(PRESETS.keys()))}")
    ap.add_argument("--presets", nargs="+", default=None,
                    help="Process multiple presets sequentially (outputs to --out_dir/preset.jsonl)")

    # Вариант 2: ручной
    ap.add_argument("--dataset", dest="name", type=str, default=None,
                    help="HF dataset repo id (alias: --name), e.g. mozilla-foundation/common_voice_17_0")
    ap.add_argument("--name", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--config", type=str, default=None, help="HF dataset config (language, etc.)")
    ap.add_argument("--split", type=str, default="train",
                    help="split spec, e.g. 'train' or 'train+validation+test'")

    # Кастомные колонки
    ap.add_argument("--audio_col", type=str, default=None, help="Audio column name (e.g., 'audio')")
    ap.add_argument("--text_col", type=str, default=None, help="Text column name (e.g., 'transcript')")

    ap.add_argument("--out", type=str, help="Output JSONL path (single dataset mode)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory for multiple presets (required with --presets)")

    ap.add_argument("--drop_empty", action="store_true", help="Skip rows with empty text or missing audio")
    ap.add_argument("--min-sec", type=float, default=None, help="Keep only samples with duration >= min-sec")
    ap.add_argument("--max-sec", type=float, default=None, help="Keep only samples with duration <= max-sec")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows written")

    # trust_remote_code по умолчанию ВКЛ — чтобы CV17/FLEURS и комьюнити-репозитории грузились без ошибки
    ap.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True,
                    help="Allow running dataset repo code (default: True)")
    ap.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false",
                    help="Disable running dataset repo code")

    # Явный выбор маппера (опционально)
    ap.add_argument("--mapper", type=str, default=None,
                    help=f"Force mapper: {', '.join(MAPPER_FUNCS.keys())}.")

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


def iter_examples(ds):
    """Универсальный итератор по датасету (Dataset или IterableDataset)."""
    try:
        # Обычный Dataset
        for ex in ds:
            yield ex
    except TypeError:
        # На всякий (streaming) — тоже итерируемо
        for ex in ds:
            yield ex


def process_dataset(name: str, config: Optional[str], split: str, mapper, args, out_path: Path) -> bool:
    print(
        f"[INFO] Loading dataset: name='{name}', config='{config}', split='{split}', trust_remote_code={args.trust_remote_code}"
    )
    try:
        ds = load_dataset(
            path=name,
            name=config,
            split=split,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset '{name}' (config={config}, split={split}).\n{e}")
        return False

    total = len(ds) if hasattr(ds, "__len__") else None
    written = 0
    skipped_empty = 0
    skipped_dur = 0

    want_min = args.min_sec is not None
    want_max = args.max_sec is not None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(iter_examples(ds)):
            row = mapper(ex)
            if row is None:
                skipped_empty += 1
                continue

            if args.drop_empty and (not row.get("audio_filepath") or not (row.get("text") or "").strip()):
                skipped_empty += 1
                continue

            if (want_min or want_max) and row.get("audio_filepath"):
                dur = compute_duration_seconds(row["audio_filepath"])
                if dur is not None:
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

            if total is not None and (i + 1) % 500 == 0:
                print(f"[PROGRESS] {i + 1}/{total} processed, {written} written...")
            elif total is None and (i + 1) % 1000 == 0:
                print(f"[PROGRESS] {i + 1} processed, {written} written...")

    print(f"[DONE] Written: {written}, skipped_empty: {skipped_empty}, skipped_dur: {skipped_dur}")
    print(f"[OUT] {out_path}")
    return True


def process_preset(preset: str, args, out_dir: Path) -> bool:
    if preset not in PRESETS:
        print(f"[SKIP] Unknown preset '{preset}'")
        return False
    p = PRESETS[preset]
    name = normalize_repo_id(p["name"])
    config = p.get("config")
    split = p.get("split") or args.split
    mapper_key = p.get("mapper")
    mapper = MAPPER_FUNCS[mapper_key] if mapper_key else MAPPER_FUNCS[choose_mapper(name, None)]
    out_path = out_dir / f"{preset.replace('-', '_')}.jsonl"
    return process_dataset(name, config, split, mapper, args, out_path)


def main():
    args = build_parser().parse_args()

    if args.presets:
        if not args.out_dir:
            raise SystemExit("--out_dir is required when using --presets")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ok_list = []
        fail_list = []
        for pr in args.presets:
            if process_preset(pr, args, out_dir):
                ok_list.append(pr)
            else:
                fail_list.append(pr)
        print(f"[SUMMARY] succeeded: {ok_list}")
        if fail_list:
            print(f"[SUMMARY] failed: {fail_list}")
        return

    if not args.out:
        raise SystemExit("--out is required when not using --presets")

    # Извлекаем параметры из пресета или ручного режима
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
            raise SystemExit("Either --preset or --dataset must be provided.")
        name = normalize_repo_id(args.name)
        config = args.config
        split = args.split
        mapper_key = None  # выберем ниже эвристикой или по колонкам

    mapper = None
    if args.audio_col and args.text_col:
        mapper = make_custom_mapper(args.audio_col, args.text_col)
    else:
        if args.mapper:
            mapper_key = args.mapper
        if not mapper_key:
            mapper_key = choose_mapper(name, forced=None)
        mapper = MAPPER_FUNCS[mapper_key]

    out_path = Path(args.out)
    success = process_dataset(name, config, split, mapper, args, out_path)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

