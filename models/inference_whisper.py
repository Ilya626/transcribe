"""Transcribe audio files using OpenAI Whisper-large-v3 model.

All model and dataset caches are forced into project-local directories so
downloads do not spill into user profile disks. CPU inference is explicitly
disallowed: CUDA GPU must be available or the script will raise.
"""
import argparse
import json
from pathlib import Path
import os
import gc
import time

import soundfile as sf
import torch

MODEL_ID = "openai/whisper-large-v3"


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
        raise RuntimeError(
            "CUDA GPU is required. CPU inference is disabled."
        )


def vram_report(tag: str) -> None:
    try:
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        free, total_rt = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated(dev)
        reserv = torch.cuda.memory_reserved(dev)
        gb = 1024 ** 3
        print(
            f"[VRAM:{tag}] alloc={alloc/gb:.2f}G reserved={reserv/gb:.2f}G free={free/gb:.2f}G total={total/gb:.2f}G"
        )
    except Exception:
        pass


def transcribe_batch(model, processor, paths: list[Path], num_beams: int, forced_decoder_ids=None) -> list[str]:
    processed = []
    for p in paths:
        audio, sr = sf.read(str(p))
        item = processor(audio, sampling_rate=sr, return_tensors="pt")
        # reduce extra batch dim for pad()
        feat = item["input_features"][0]
        processed.append({"input_features": feat})
    batch = processor.feature_extractor.pad(processed, return_tensors="pt")
    try:
        model_dtype = next(model.parameters()).dtype
    except Exception:
        model_dtype = torch.float16
    inputs = {}
    feats = batch.get("input_features")
    if hasattr(feats, "to"):
        feats = feats.to(model.device)
        if feats.dtype != model_dtype:
            feats = feats.to(model_dtype)
    inputs["input_features"] = feats
    if "attention_mask" in batch and hasattr(batch["attention_mask"], "to"):
        inputs["attention_mask"] = batch["attention_mask"].to(model.device)
    gen_kwargs = {"num_beams": int(num_beams)}
    if forced_decoder_ids is not None:
        gen_kwargs["forced_decoder_ids"] = forced_decoder_ids
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [t if isinstance(t, str) else str(t) for t in texts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to an audio file or directory")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    parser.add_argument("--model_id", default=MODEL_ID, help="HF model id to use")
    # Defaults per model (tuned on 12GB GPU, 200 clips)
    # large-v3: bs≈24; large-v3-turbo: bs≈64; distil-large-v3.5: bs≈64.
    def _default_bs(mid: str) -> int:
        m = (mid or '').lower()
        if 'large-v3-turbo' in m or 'v3-turbo' in m or (('turbo' in m) and ('whisper' in m)):
            return 64
        if 'distil' in m and 'large' in m:
            return 64
        if 'large-v3' in m:
            return 24
        # Fallback for other Whisper sizes
        return 16
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width")
    parser.add_argument("--language", type=str, default="ru", help="Language code for Whisper (e.g., ru)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Whisper task")
    parser.add_argument("--engine", type=str, default="generate", choices=["generate", "pipeline"], help="Inference engine")
    args = parser.parse_args()

    configure_local_caches()
    require_cuda()

    # Import after env configured to ensure caches respected
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from transformers import pipeline as hf_pipeline

    # Simple inter-process GPU lock to avoid parallel runs on same GPU
    repo_root = Path(__file__).resolve().parents[1]
    lock_dir = repo_root / ".tmp"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "gpu.lock"
    pid = os.getpid()
    lock_acquired = False
    for attempt in range(120):  # up to ~2 minutes
        try:
            # O_EXCL ensures single owner
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(pid))
            lock_acquired = True
            print(f"[LOCK] Acquired GPU lock at {lock_path} (pid={pid})")
            break
        except FileExistsError:
            try:
                holder = (lock_path.read_text(encoding="utf-8").strip())
            except Exception:
                holder = "unknown"
            print(f"[LOCK] Waiting for GPU lock held by pid={holder} ...")
            time.sleep(1)
    if not lock_acquired:
        raise RuntimeError(f"Could not acquire GPU lock at {lock_path}; another process is running")

    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )
    model.to(device).eval()
    # Configure generation for language/task explicitly to avoid auto language detection/translation
    try:
        # Prefer explicit prompt over autodetect
        gen = model.generation_config
        gen.language = args.language
        gen.task = args.task
        # If model/config carries stale forced ids, clear them and set from processor
        try:
            model.config.forced_decoder_ids = None
        except Exception:
            pass
        try:
            prompt_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
        except Exception:
            prompt_ids = None
    except Exception:
        prompt_ids = None
    if args.batch_size is None:
        args.batch_size = _default_bs(args.model_id)
    vram_report("loaded")

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".jsonl" and input_path.is_file():
        audio_files = []
        # Handle potential UTF-8 BOM with 'utf-8-sig'
        with open(input_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                p = obj.get("audio_filepath") or obj.get("audio")
                if p:
                    audio_files.append(Path(p))
    else:
        audio_files = [input_path] if input_path.is_file() else sorted(
            p for p in input_path.glob("**/*") if p.suffix.lower() in {".wav", ".flac", ".mp3"}
        )

    results = {}
    bs = max(1, int(args.batch_size))
    try:
        if args.engine == "pipeline":
            asr = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=0,
                torch_dtype=next(model.parameters()).dtype,
                generate_kwargs={"task": args.task, "language": args.language},
                chunk_length_s=30,
            )
            for i in range(0, len(audio_files), bs):
                batch_paths = audio_files[i : i + bs]
                print(f"Transcribing batch {i//bs+1} [{len(batch_paths)} files] via pipeline...")
                texts = []
                for p in batch_paths:
                    o = asr(str(p), return_timestamps=False)
                    texts.append(o.get("text", "") if isinstance(o, dict) else (o[0].get("text", "") if (isinstance(o, list) and o) else ""))
                for p, t in zip(batch_paths, texts):
                    results[str(p)] = t
                vram_report(f"batch_{i//bs+1}")
                del texts
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            for i in range(0, len(audio_files), bs):
                batch_paths = audio_files[i : i + bs]
                print(f"Transcribing batch {i//bs+1} [{len(batch_paths)} files] ...")
                # Force language/task via decoder prompt, if available
                texts = transcribe_batch(model, processor, batch_paths, args.num_beams, forced_decoder_ids=prompt_ids)
                for p, t in zip(batch_paths, texts):
                    results[str(p)] = t
                vram_report(f"batch_{i//bs+1}")
                # Encourage timely release of temps
                del texts
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    finally:
        # Explicit cleanup to avoid lingering allocations when used programmatically
        try:
            del model
            del processor
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        vram_report("final")
        # Release GPU lock
        try:
            if lock_acquired and lock_path.exists():
                # Remove only if still ours
                content = ""
                try:
                    content = lock_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
                if content == str(pid):
                    lock_path.unlink(missing_ok=True)
                    print(f"[LOCK] Released GPU lock at {lock_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
