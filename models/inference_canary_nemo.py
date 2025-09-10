"""Inference with NVIDIA Canary (NeMo .nemo checkpoint).

Loads the Canary 1B v2 NeMo model and transcribes audio from a file, directory,
JSON/JSONL manifest (with keys `audio_filepath` or `audio`). Stores results as
JSON mapping `path -> text`.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import gc


def configure_local_caches():
    repo_root = Path(__file__).resolve().parents[1]
    hf = os.environ.get("HF_HOME") or str(repo_root / ".hf")
    os.environ.setdefault("HF_HOME", hf)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(hf) / "hub"))
    os.environ.setdefault("TORCH_HOME", str(repo_root / ".torch"))
    tmp = str(repo_root / ".tmp")
    os.environ.setdefault("TMP", tmp)
    os.environ.setdefault("TEMP", tmp)
    for d in [hf, os.environ["HF_HUB_CACHE"], os.environ["TORCH_HOME"], tmp]:
        Path(d).mkdir(parents=True, exist_ok=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. CPU inference is disabled.")


def load_nemo_model(model_id: str, nemo_path: str | None) -> "EncDecMultiTaskModel":
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    from nemo.collections.asr.models import EncDecMultiTaskModel

    if nemo_path:
        path = nemo_path
    else:
        # Assume standard filename inside the repo: <last-token>.nemo
        fname = (model_id.split("/")[-1]).strip() + ".nemo"
        # Download into local HF cache directory for this repo
        local_dir = str(Path(os.environ.get("HF_HOME", ".")) / f"models--{model_id.replace('/', '--')}")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename=fname,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=token,
            )
        except Exception as e:
            # Fallback: direct HTTP download via requests
            url = f"https://huggingface.co/{model_id}/resolve/main/{fname}"
            dest = Path(local_dir) / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                import requests

                headers = {"Authorization": f"Bearer {token}"} if token else {}
                with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                path = str(dest)
            except Exception as e2:
                raise RuntimeError(
                    f"Unable to download {model_id} via hf_hub or direct HTTP: {e2}"
                ) from e2
    model = EncDecMultiTaskModel.restore_from(path, map_location="cuda")
    model.eval()
    return model


def collect_audio(input_path: Path) -> List[Path]:
    exts = {".wav", ".flac", ".mp3"}
    if input_path.is_file():
        if input_path.suffix.lower() == ".jsonl":
            out: List[Path] = []
            with input_path.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    p = o.get("audio_filepath") or o.get("audio")
                    if p:
                        out.append(Path(p))
            return out
        if input_path.suffix.lower() == ".json":
            with input_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return [Path(k) for k in data.keys()]
            elif isinstance(data, list):
                out: List[Path] = []
                for o in data:
                    if isinstance(o, dict):
                        p = o.get("audio_filepath") or o.get("audio")
                        if p:
                            out.append(Path(p))
                return out
        return [input_path]
    else:
        return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in exts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=str, help="Path to audio file/dir or JSON/JSONL manifest")
    ap.add_argument("output", type=str, help="Path to output JSON file")
    ap.add_argument("--model_id", default="nvidia/canary-1b-v2", help="HF repo id with .nemo file")
    ap.add_argument("--nemo", default=None, help="Path to local .nemo file (overrides --model_id)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--source_lang", default="ru", help="Source language code (e.g., ru)")
    ap.add_argument("--target_lang", default="ru", help="Target language code (use same as source for transcription)")
    ap.add_argument("--task", default="asr", choices=["asr", "translate"], help="Task: asr (transcribe) or translate")
    ap.add_argument("--pnc", default="yes", choices=["yes", "no"], help="Include punctuation (yes/no)")
    args = ap.parse_args()

    configure_local_caches()
    require_cuda()

    model = load_nemo_model(args.model_id, args.nemo)

    audio_files = collect_audio(Path(args.input))
    results: dict[str, str] = {}

    # Use model.transcribe on batches of file paths
    bs = max(1, int(args.batch_size))
    def _to_text(h) -> str:
        try:
            if isinstance(h, str):
                return h
            if isinstance(h, dict):
                return h.get("text") or h.get("pred_text") or h.get("transcription") or next((v for v in h.values() if isinstance(v, str)), "")
            # Hypothesis-like objects
            for attr in ("text", "pred_text", "transcription", "answer"):
                if hasattr(h, attr):
                    v = getattr(h, attr)
                    if isinstance(v, str):
                        return v
            return str(h)
        except Exception:
            return ""

    def vram_report(tag: str) -> None:
        try:
            dev = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(dev).total_memory
            free, _total_rt = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated(dev)
            reserv = torch.cuda.memory_reserved(dev)
            gb = 1024 ** 3
            print(f"[VRAM:{tag}] alloc={alloc/gb:.2f}G reserved={reserv/gb:.2f}G free={free/gb:.2f}G total={total/gb:.2f}G")
        except Exception:
            pass

    for i in range(0, len(audio_files), bs):
        batch = audio_files[i : i + bs]
        print(f"Transcribing batch {i//bs+1} [{len(batch)} files] ...")
        paths = [str(p) for p in batch]
        hyps = None
        try:
            hyps = model.transcribe(
                paths,
                batch_size=bs,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                task=args.task,
                pnc=args.pnc,
            )
            for p, h in zip(batch, hyps):
                results[str(p)] = _to_text(h)
        finally:
            # Aggressive cleanup between batches to avoid lingering allocations
            try:
                del hyps
                del paths
            except Exception:
                pass
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            vram_report(f"batch_{i//bs+1}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
