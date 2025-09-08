"""Transcribe audio using Salute GigaAM models.

Brings CLI/UX closer to Whisper script:
- Project-local caches, GPU-only policy, manifest/file/dir input.
- Batch-style loop with VRAM reporting and a simple GPU lock.
"""
import argparse
import json
from pathlib import Path
import os
import gc
import time
import subprocess
import sys

import soundfile as sf  # keep dependency present for potential pre-processing
try:
    import gigaam  # type: ignore
except Exception as e:
    gigaam = None  # type: ignore
    _IMPORT_ERR = e
import torch
import math
import tempfile
import soundfile as sf


def configure_local_caches():
    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TORCH_HOME", str(repo_root / ".torch"))
    tmp = str(repo_root / ".tmp")
    os.environ.setdefault("TMP", tmp)
    os.environ.setdefault("TEMP", tmp)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(tmp).mkdir(parents=True, exist_ok=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. CPU inference is disabled.")


def vram_report(tag: str) -> None:
    try:
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        free, total_rt = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated(dev)
        reserv = torch.cuda.memory_reserved(dev)
        gb = 1024 ** 3
        print(f"[VRAM:{tag}] alloc={alloc/gb:.2f}G reserved={reserv/gb:.2f}G free={free/gb:.2f}G total={total/gb:.2f}G")
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


def _chunk_transcribe(model, path: Path, chunk_sec: float = 15.0, overlap_sec: float = 1.0) -> str:
    try:
        audio, sr = sf.read(str(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        n = len(audio)
        hop = int(max(1, (chunk_sec - overlap_sec) * sr))
        win = int(max(1, chunk_sec * sr))
        parts = []
        tmpdir = Path(tempfile.gettempdir()) / "gigaam_chunks"
        tmpdir.mkdir(parents=True, exist_ok=True)
        idx = 0
        pos = 0
        while pos < n:
            seg = audio[pos: pos + win]
            if len(seg) == 0:
                break
            tmp_wav = tmpdir / f"chunk_{path.stem}_{idx}.wav"
            sf.write(str(tmp_wav), seg, sr)
            try:
                with torch.inference_mode():
                    t = model.transcribe(str(tmp_wav))
            finally:
                try:
                    tmp_wav.unlink(missing_ok=True)
                except Exception:
                    pass
            if isinstance(t, dict):
                t = t.get("transcription") or t.get("text") or ""
            parts.append(t if isinstance(t, str) else str(t))
            idx += 1
            pos += hop
        return " ".join(parts).strip()
    except Exception as e:  # pragma: no cover - last resort
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to audio file/dir or JSONL/JSON manifest")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    parser.add_argument(
        "--model", default="v2_rnnt", help="Model type: v2_rnnt, v2_ctc, rnnt, ctc, etc."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Files per loop batch (I/O grouping)")
    parser.add_argument("--sample", type=int, default=0, help="Sample N files from manifest (0=all)")
    parser.add_argument("--no_lock", action="store_true", help="Do not acquire GPU lock (use for deliberate multi-process runs)")
    parser.add_argument("--shard_idx", type=int, default=0, help="This process shard index (0..shard_total-1)")
    parser.add_argument("--shard_total", type=int, default=1, help="Total number of shards (>=1). Shards are taken by round-robin index")
    parser.add_argument("--parallel", type=int, default=1, help="Launch N parallel shards (parent orchestrates and merges)")
    parser.add_argument("--child", action="store_true", help="Mark this process as a child (internal use)")
    args = parser.parse_args()

    configure_local_caches()
    require_cuda()

    if gigaam is None:
        raise SystemExit(
            "GigaAM is not installed. On Windows Python 3.13, sentencepiece wheels are unavailable.\n"
            "Use Python 3.11/3.12 venv and then: pip install gigaam  (or git+https://github.com/salute-developers/GigaAM).\n"
            f"Underlying import error: {_IMPORT_ERR}"
        )

    # Orchestrate parallel launch if requested (parent process)
    if args.parallel and args.parallel > 1 and not args.child:
        total = int(args.parallel)
        script_path = Path(__file__).resolve()
        out_base = Path(args.output)
        out_dir = out_base.parent
        out_stem = out_base.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        procs = []
        for i in range(total):
            shard_out = out_dir / f"{out_stem}_shard_{i}.json"
            cmd = [
                sys.executable,
                str(script_path),
                str(args.input),
                str(shard_out),
                "--model", str(args.model),
                "--batch_size", str(args.batch_size),
                "--no_lock",
                "--shard_idx", str(i),
                "--shard_total", str(total),
                "--child",
            ]
            print(f"[PARENT] Launching shard {i}/{total}: {cmd}")
            procs.append(subprocess.Popen(cmd))
        rc = [p.wait() for p in procs]
        print(f"[PARENT] Shards exited: {rc}")
        # Merge
        merged: dict[str, str] = {}
        for i in range(total):
            shard_out = out_dir / f"{out_stem}_shard_{i}.json"
            try:
                with shard_out.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                merged.update({str(k): str(v) for k, v in d.items()})
            except Exception as e:
                print(f"[PARENT][WARN] cannot read {shard_out}: {e}")
        with out_base.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[PARENT] Merged {len(merged)} items -> {out_base}")
        return

    # GPU lock (optional)
    repo_root = Path(__file__).resolve().parents[1]
    lock_path = repo_root / ".tmp" / "gpu.lock"
    pid = os.getpid()
    lock_acquired = False
    if not args.no_lock:
        lock_acquired, pid = acquire_gpu_lock(lock_path)
        if not lock_acquired:
            raise RuntimeError(f"Could not acquire GPU lock at {lock_path}; another process is running")

    model = gigaam.load_model(args.model)
    if hasattr(model, "eval"):
        model.eval()
    try:
        if hasattr(model, "to"):
            model = model.to("cuda")
        elif hasattr(model, "cuda"):
            model = model.cuda()
    except Exception:
        pass

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

    # Optional sampling for quick checks
    if args.sample and args.sample > 0 and len(audio_files) > args.sample:
        import random
        random.seed(42)
        random.shuffle(audio_files)
        audio_files = audio_files[: args.sample]

    # Sharding for multi-process launches (round-robin by index)
    if args.shard_total and args.shard_total > 1:
        keep = []
        for idx, p in enumerate(audio_files):
            if (idx % args.shard_total) == max(0, int(args.shard_idx)):
                keep.append(p)
        print(f"[SHARD] idx={args.shard_idx} of total={args.shard_total} -> {len(keep)} files")
        audio_files = keep

    results: dict[str, str] = {}
    bs = max(1, int(args.batch_size))
    try:
        for i in range(0, len(audio_files), bs):
            batch_paths = audio_files[i : i + bs]
            print(f"Transcribing batch {i//bs+1} [{len(batch_paths)} files] ...")
            for p in batch_paths:
                txt = ""
                try:
                    with torch.inference_mode():
                        transcription = model.transcribe(str(p))
                except Exception as e:
                    # Fallback for long-form audios
                    msg = str(e).lower()
                    if "longform" in msg and hasattr(model, "transcribe_longform"):
                        try:
                            with torch.inference_mode():
                                transcription = model.transcribe_longform(str(p))
                        except Exception:
                            # As a last resort, do naive chunking without VAD
                            transcription = _chunk_transcribe(model, Path(p))
                    else:
                        raise
                if isinstance(transcription, dict):
                    txt = (
                        transcription.get("transcription")
                        or transcription.get("text")
                        or ""
                    )
                else:
                    txt = transcription if isinstance(transcription, str) else str(transcription)
                results[str(p)] = txt
            vram_report(f"batch_{i//bs+1}")
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    finally:
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        vram_report("final")
        if lock_acquired:
            release_gpu_lock(lock_path, pid)


if __name__ == "__main__":
    main()
