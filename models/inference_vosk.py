"""Transcribe audio files using Vosk offline model.

Keeps temporary files within the project directory.

Note: Vosk is a CPU-only recognizer in typical setups. Since CPU inference is
disallowed by policy, this script will raise unless override is implemented.
"""
import argparse
import json
from pathlib import Path
import wave
import os

from vosk import Model, KaldiRecognizer


def transcribe_file(model: Model, path: Path) -> str:
    with wave.open(str(path), "rb") as wf:
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)
        result = json.loads(rec.FinalResult())
    return result.get("text", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=str, help="Path to Vosk model directory")
    parser.add_argument("input", type=str, help="Path to an audio file or directory of WAV files")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tmp = str(repo_root / ".tmp")
    os.environ.setdefault("TMP", tmp)
    os.environ.setdefault("TEMP", tmp)
    Path(tmp).mkdir(parents=True, exist_ok=True)

    raise RuntimeError(
        "CPU-only Vosk is not permitted under GPU-only policy. Use Whisper/Canary/Silero/GigaAM."
    )

    input_path = Path(args.input)
    audio_files = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.glob("**/*") if p.suffix.lower() == ".wav"
    )

    results = {}
    for audio_path in audio_files:
        print(f"Transcribing {audio_path}...")
        results[str(audio_path)] = transcribe_file(model, audio_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
