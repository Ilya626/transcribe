"""Transcribe audio files using Silero STT model.

Ensures Torch Hub cache is inside the project directory. Requires CUDA; CPU
inference is disabled.
"""
import argparse
import json
from pathlib import Path
import os

import torch


def load_model(device):
    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TORCH_HOME", str(repo_root / ".torch"))
    tmp = str(repo_root / ".tmp")
    os.environ.setdefault("TMP", tmp)
    os.environ.setdefault("TEMP", tmp)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(tmp).mkdir(parents=True, exist_ok=True)
    if device != "cuda":
        raise RuntimeError("CUDA GPU is required. CPU inference is disabled.")
    model, decoder, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_stt",
        language="ru",
        model_id="ru_v4",
    )
    (read_batch, split_into_batches, read_audio, prepare_model_input) = utils
    model.to(device).eval()
    return model, decoder, read_audio, split_into_batches, prepare_model_input


def transcribe_file(model, decoder, read_audio, split_into_batches, prepare_model_input, path: Path) -> str:
    audio = read_audio(str(path))
    batches = split_into_batches([audio], batch_size=1)
    input_data = prepare_model_input(batches, device=model.device)
    with torch.inference_mode():
        output = model(input_data)
    transcription = decoder(output[0].cpu())
    return transcription


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to an audio file or directory")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. CPU inference is disabled.")
    device = "cuda"
    model, decoder, read_audio, split_into_batches, prepare_model_input = load_model(device)

    input_path = Path(args.input)
    audio_files = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.glob("**/*") if p.suffix.lower() in {".wav", ".flac", ".mp3"}
    )

    results = {}
    for audio_path in audio_files:
        print(f"Transcribing {audio_path}...")
        results[str(audio_path)] = transcribe_file(
            model, decoder, read_audio, split_into_batches, prepare_model_input, audio_path
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
