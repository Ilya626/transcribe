"""
RU speech dataset presets and helpers for HF-based manifest building.

Each preset provides one or more candidate (dataset_id, config, split, columns)
tuples. The builder will try candidates in order until a load succeeds. This is
to tolerate small ID/config drift across HF mirrors.

Notes:
- These presets are best-effort. If HF IDs change, override on CLI.
- Audio column is typically "audio" (HF Datasets Audio feature), and text is
  one of: sentence/transcript/transcription/text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class Candidate:
    dataset: str
    config: Optional[str]
    split: str
    audio_col: str
    text_col: str


PRESETS: dict[str, List[Candidate]] = {
    # Common Voice v17 Russian
    "cv17-ru": [
        Candidate(
            dataset="mozilla-foundation/common_voice_17_0",
            config="ru",
            split="train+validation+test",
            audio_col="audio",
            text_col="sentence",
        ),
    ],
    # Multilingual LibriSpeech Russian
    "mls-ru": [
        Candidate(
            dataset="mls",
            config="ru",
            split="train+dev+test",
            audio_col="audio",
            text_col="transcript",
        ),
        Candidate(
            dataset="facebook/multilingual_librispeech",
            config="ru",
            split="train+dev+test",
            audio_col="audio",
            text_col="transcript",
        ),
    ],
    # FLEURS Russian (locale ru_ru)
    "fleurs-ru": [
        Candidate(
            dataset="google/fleurs",
            config="ru_ru",
            split="train+validation+test",
            audio_col="audio",
            text_col="transcription",
        ),
    ],
    # Russian LibriSpeech (community mirrors vary). Try common aliases.
    "ruls": [
        Candidate(
            dataset="bond005/russian_librispeech",
            config=None,
            split="train+validation+test",
            audio_col="audio",
            text_col="text",
        ),
        Candidate(
            dataset="Yehor/russian_librispeech",
            config=None,
            split="train+validation+test",
            audio_col="audio",
            text_col="text",
        ),
        Candidate(
            dataset="Russian-Librispeech/russian_librispeech",
            config=None,
            split="train+validation+test",
            audio_col="audio",
            text_col="text",
        ),
    ],
    # GOLOS (crowd and farfield) — IDs vary; provide typical mirrors.
    "golos-crowd": [
        Candidate(
            dataset="sberdevices/golos",
            config="crowd",
            split="train+validation+test",
            audio_col="audio",
            text_col="text",
        ),
        Candidate(
            dataset="bond005/golos-crowd",
            config=None,
            split="train+validation+test",
            audio_col="audio",
            text_col="text",
        ),
    ],
    "golos-farfield": [
        Candidate(
            dataset="sberdevices/golos",
            config="farfield",
            split="train+validation+test",
            audio_col="audio",
            text_col="text",
        ),
        Candidate(
            dataset="bond005/golos-farfield",
            config=None,
            split="train+validation+test",
            audio_col="audio",
            text_col="text",
        ),
    ],
    # Podlodka Speech (placeholder; adjust if your HF ID differs)
    "podlodka": [
        Candidate(
            dataset="podlodka/speech",
            config=None,
            split="train+validation+test",
            audio_col="audio",
            text_col="text",
        ),
    ],
    # Non-speech noise fragments (optional; use carefully in training)
    "audioset-nonspeech": [
        Candidate(
            dataset="speechcolab/audioset-nonspeech",
            config=None,
            split="train+validation+test",
            audio_col="audio",
            text_col="label",  # will be ignored; we will set empty text
        ),
    ],
}


def list_presets() -> Iterable[str]:
    return PRESETS.keys()


