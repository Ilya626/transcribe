import argparse
import json
import string
from pathlib import Path
from typing import List, Dict, Tuple

import soundfile as sf
from jiwer import cer, wer


def normalize(text: str) -> str:
    """Lower-case and strip punctuation."""
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def load_json(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def segment_metrics(ref: str, pred: str) -> Tuple[float, float]:
    ref_norm = normalize(ref)
    pred_norm = normalize(pred)
    return wer(ref_norm, pred_norm), cer(ref_norm, pred_norm)


def audio_duration(path: Path) -> float:
    """Return duration of an audio file in seconds."""
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def select_segments(
    references: Dict[str, str],
    predictions: Dict[str, str],
    hours: float,
) -> List[Dict[str, object]]:
    """Return list of segments sorted by WER until reaching the given hours."""
    segments = []
    for audio_path, ref in references.items():
        if audio_path not in predictions:
            continue
        pred = predictions[audio_path]
        w, c = segment_metrics(ref, pred)
        dur = audio_duration(Path(audio_path))
        segments.append(
            {
                "audio": audio_path,
                "reference": ref,
                "prediction": pred,
                "wer": w,
                "cer": c,
                "duration": dur,
            }
        )
    segments.sort(key=lambda x: x["wer"], reverse=True)
    target = hours * 3600
    total = 0.0
    selected = []
    for seg in segments:
        if total >= target:
            break
        selected.append(seg)
        total += seg["duration"]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select hardest segments for fine-tuning based on WER"
    )
    parser.add_argument("references", type=str, help="Path to reference transcripts JSON")
    parser.add_argument("predictions", type=str, help="Path to predictions JSON")
    parser.add_argument(
        "output", type=str, help="Where to store selected segments (JSON)"
    )
    parser.add_argument(
        "--hours", type=float, default=8.0, help="Total hours of audio to select"
    )
    args = parser.parse_args()

    refs = load_json(Path(args.references))
    preds = load_json(Path(args.predictions))

    selected = select_segments(refs, preds, args.hours)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    total_hours = sum(seg["duration"] for seg in selected) / 3600
    print(
        f"Selected {len(selected)} segments totaling {total_hours:.2f} hours of audio"
    )


if __name__ == "__main__":
    main()
