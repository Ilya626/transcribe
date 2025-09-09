"""
python transcribe/training/runpod_nemo_canary_lora.py
--auto_download --model_id nvidia/canary-1b-v2
--nemo /workspace/models/canary-1b-v2.nemo
--train /workspace/data/train_portable.jsonl
--val /workspace/data/val_portable.jsonl
--outdir /workspace/exp/canary_ru_lora_a6000
--export /workspace/models/canary-ru-lora-a6000.nemo
--bs 8 --accum 1 --precision bf16 --num_workers 8
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05
#!/usr/bin/env python

Fetch a dataset archive into /workspace (or current repo) and extract it.

Supports two simple sources:
- Hugging Face private/public repo (requires HF_TOKEN if private)
- Direct URL (HTTP/HTTPS)

Examples (Runpod):
  # Hugging Face (private dataset repo)
  export HF_TOKEN=hf_xxx
  python transcribe/tools/fetch_dataset.py \
    --hf_repo your-org/your-dataset \
    --hf_file dataset_portable.tgz \
    --dest /workspace/data

  # Direct URL (e.g. from transfer.sh or S3 presigned URL)
  python transcribe/tools/fetch_dataset.py \
    --url https://transfer.sh/abc123/dataset_portable.tgz \
    --dest /workspace/data

The script will:
- download the archive to <dest>/dataset_archive
- auto-extract .tgz/.tar.gz/.zip
- print the extracted paths
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile


def download_hf(repo: str, filename: str, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        raise SystemExit("huggingface_hub not installed. pip install huggingface_hub\n" + str(e))
    path = hf_hub_download(repo_id=repo, filename=filename, token=token)
    return Path(path)


def download_url(url: str, out_path: Path) -> Path:
    import urllib.request
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:  # nosec B310
        shutil.copyfileobj(r, f)
    return out_path


def auto_extract(archive: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    low = archive.name.lower()
    if low.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)  # nosec B202
            out = [dest / m.name for m in tf.getmembers()]
    elif low.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
            out = [dest / n for n in zf.namelist()]
    else:
        out = [archive]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf_repo", default="", help="HF repo id (org/name)")
    ap.add_argument("--hf_file", default="", help="Filename in HF repo (e.g., dataset_portable.tgz)")
    ap.add_argument("--url", default="", help="Direct URL to archive")
    ap.add_argument("--dest", default="/workspace/data", help="Destination directory")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    archive_path: Path
    if args.hf_repo and args.hf_file:
        token = os.environ.get("HF_TOKEN")
        print(f"[HF] Downloading {args.hf_repo}:{args.hf_file} ...")
        cache_path = download_hf(args.hf_repo, args.hf_file, token)
        archive_path = dest / Path(args.hf_file).name
        try:
            shutil.copyfile(cache_path, archive_path)
        except Exception:
            archive_path = Path(cache_path)
        print(f"[HF] Saved archive -> {archive_path}")
    elif args.url:
        name = os.path.basename(args.url).split("?")[0] or "dataset_portable.tgz"
        archive_path = dest / name
        print(f"[URL] Downloading {args.url} -> {archive_path}")
        download_url(args.url, archive_path)
    else:
        raise SystemExit("Provide either --hf_repo + --hf_file or --url")

    extracted = auto_extract(archive_path, dest)
    print("[OK] Extracted entries:")
    for p in extracted[:20]:
        print("  ", p)
    if len(extracted) > 20:
        print(f"  ... and {len(extracted)-20} more")


if __name__ == "__main__":
    main()

