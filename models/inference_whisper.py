"""Transcribe audio files using OpenAI Whisper-large-v3 model."""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "openai/whisper-large-v3"


def transcribe_file(model, processor, path: Path) -> str:
    audio, sr = sf.read(str(path))
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to an audio file or directory")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
    model.to(device).eval()

    input_path = Path(args.input)
    audio_files = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.glob("**/*") if p.suffix.lower() in {".wav", ".flac", ".mp3"}
    )

    results = {}
    for audio_path in audio_files:
        print(f"Transcribing {audio_path}...")
        results[str(audio_path)] = transcribe_file(model, processor, audio_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
