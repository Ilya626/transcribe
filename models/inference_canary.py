"""Transcribe audio using NVIDIA Canary models via Hugging Face Transformers.

Features aligned with Whisper script:
- Project-local caches; GPU-only policy with clear error if no CUDA.
- Manifest input (JSONL with audio_filepath/audio) or file/dir of audio.
- Batch processing with default batch size heuristic.
- Optional language/task forcing via decoder prompt ids.
- Simple inter-process GPU lock to avoid parallel runs on same GPU.
- VRAM usage reports per batch and at start/end.
"""
import argparse
import json
from pathlib import Path
import os
import gc
import time

import soundfile as sf
import torch

MODEL_ID = "nvidia/canary-1b-v2"


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


def acquire_gpu_lock(lock_path: Path, timeout_s: int = 120) -> tuple[bool, int]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(pid))
            print(f"[LOCK] Acquired GPU lock at {lock_path}")
            return True, pid
        except FileExistsError:
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return False, pid


def release_gpu_lock(lock_path: Path, owner_pid: int) -> None:
    try:
        if lock_path.exists():
            try:
                content = lock_path.read_text(encoding="utf-8").strip()
            except Exception:
                content = ""
            if content == str(owner_pid):
                lock_path.unlink(missing_ok=True)
                print(f"[LOCK] Released GPU lock at {lock_path}")
    except Exception:
        pass


def transcribe_batch(model, processor, paths: list[Path], num_beams: int, forced_decoder_ids=None) -> list[str]:
    processed = []
    for p in paths:
        audio, sr = sf.read(str(p))
        item = processor(audio, sampling_rate=sr, return_tensors="pt")
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
    parser.add_argument("input", type=str, help="Path to audio file/dir or JSONL/JSON manifest")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    parser.add_argument("--model_id", default=MODEL_ID, help="HF model id to use")

    def _default_bs(mid: str) -> int:
        m = (mid or '').lower()
        if 'canary' in m:
            return 32
        return 16

    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width")
    parser.add_argument("--language", type=str, default="ru", help="Language code (if supported)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Task for decoding")
    parser.add_argument("--engine", type=str, default="generate", choices=["generate", "pipeline"], help="Inference engine")
    args = parser.parse_args()

    configure_local_caches()
    require_cuda()

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from transformers import pipeline as hf_pipeline

    # GPU lock
    repo_root = Path(__file__).resolve().parents[1]
    lock_path = repo_root / ".tmp" / "gpu.lock"
    lock_acquired, pid = acquire_gpu_lock(lock_path)
    if not lock_acquired:
        raise RuntimeError(f"Could not acquire GPU lock at {lock_path}; another process is running")

    device = "cuda"
    use_pipeline_only = False
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id, language=args.language, task=args.task, trust_remote_code=True
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_id, torch_dtype=torch.float16, trust_remote_code=True
        )
        model.to(device).eval()
    except Exception as e:
        print(f"[WARN] Failed to init model/processor directly: {e}\n        Falling back to transformers.pipeline with trust_remote_code.")
        use_pipeline_only = True

    # Configure generation for language/task explicitly
    prompt_ids = None
    if not use_pipeline_only:
        try:
            gen = model.generation_config
            gen.language = args.language
            gen.task = args.task
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
    if input_path.suffix.lower() in {".jsonl", ".json"} and input_path.is_file():
        audio_files: list[Path] = []
        if input_path.suffix.lower() == ".jsonl":
            with open(input_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    p = obj.get("audio_filepath") or obj.get("audio")
                    if p:
                        audio_files.append(Path(p))
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                audio_files = [Path(k) for k in obj.keys()]
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        p = it.get("audio_filepath") or it.get("audio")
                        if p:
                            audio_files.append(Path(p))
    else:
        audio_files = [input_path] if input_path.is_file() else sorted(
            p for p in input_path.glob("**/*") if p.suffix.lower() in {".wav", ".flac", ".mp3"}
        )

    results: dict[str, str] = {}
    bs = max(1, int(args.batch_size))
    try:
        if args.engine == "pipeline" or use_pipeline_only:
            asr = hf_pipeline(
                "automatic-speech-recognition",
                model=(model if not use_pipeline_only else args.model_id),
                tokenizer=(None if use_pipeline_only else getattr(processor, "tokenizer", None)),
                feature_extractor=(None if use_pipeline_only else getattr(processor, "feature_extractor", None)),
                device=0,
                torch_dtype=(next(model.parameters()).dtype if not use_pipeline_only else torch.float16),
                trust_remote_code=True,
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
                texts = transcribe_batch(model, processor, batch_paths, args.num_beams, forced_decoder_ids=prompt_ids)
                for p, t in zip(batch_paths, texts):
                    results[str(p)] = t
                vram_report(f"batch_{i//bs+1}")
                del texts
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    finally:
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
        release_gpu_lock(lock_path, pid)


if __name__ == "__main__":
    main()
