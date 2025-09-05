"""Transcribe audio files using GigaAM v2 model."""
import argparse
import json
from pathlib import Path

import soundfile as sf  # ensures dependency available if audio pre-processing needed
import gigaam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to an audio file or directory")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    parser.add_argument(
        "--model", default="v2_rnnt", help="Model type: v2_rnnt, v2_ctc, rnnt, ctc, etc."
    )
    args = parser.parse_args()

    model = gigaam.load_model(args.model)

    input_path = Path(args.input)
    audio_files = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.glob("**/*") if p.suffix.lower() in {".wav", ".flac", ".mp3"}
    )

    results = {}
    for audio_path in audio_files:
        print(f"Transcribing {audio_path}...")
        transcription = model.transcribe(str(audio_path))
        # some versions may return dict with 'transcription' key
        if isinstance(transcription, dict):
            transcription = transcription.get("transcription") or transcription.get("text") or ""
        results[str(audio_path)] = transcription

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
