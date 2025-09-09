#!/usr/bin/env python
"""Build the RU Stage-1 dataset pipeline.

Fetch HF datasets, convert them to manifests, mix Golos parts,
then assemble the final Stage-1 training mix using methodology ratios.

This script orchestrates the following steps:
1. `build_manifest_hf` for Common Voice, Russian LibriSpeech,
   Podlodka, Golos crowd/farfield (and optional non-speech).
2. Quality filter each manifest via Canary (`filter_manifest_canary`).
3. Concatenate Golos crowd+farfield into one manifest.
4. Invoke `build_ru_stage1_mix` to sample the final training mix.

Telephony data has no HF preset, so provide its manifest via `--telephony`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run(cmd: List[str], cwd: Path) -> None:
    """Run a subprocess in ``cwd``, echoing the command."""
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="data", help="Directory for manifests and final mix")
    ap.add_argument("--telephony", required=True, help="Path to telephony manifest JSONL")
    ap.add_argument("--target_size", type=int, default=1_000_000, help="Rows in final mix")
    ap.add_argument("--include_nonspeech", action="store_true", help="Include AudioSet non-speech preset")
    ap.add_argument("--seed", type=int, default=42, help="Seed for shuffling")
    ap.add_argument("--max_wer", type=float, default=0.15, help="Quality filter max WER for Canary")
    ap.add_argument("--min_dur", type=float, default=1.0, help="Quality filter min duration (s)")
    ap.add_argument("--max_dur", type=float, default=35.0, help="Quality filter max duration (s)")
    ap.add_argument("--batch_size", type=int, default=None, help="Batch size for Canary inference")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    repo_root = Path(__file__).resolve().parents[2]

    def bm(preset: str, out_path: Path) -> None:
        cmd = [
            py,
            "-m",
            "transcribe.tools.build_manifest_hf",
            "--preset",
            preset,
            "--out",
            str(out_path),
            "--drop_empty",
        ]
        run(cmd, cwd=repo_root)

    def filt(manifest: Path, out_path: Path) -> Path:
        cmd = [
            py,
            "-m",
            "transcribe.tools.filter_manifest_canary",
            "--manifest",
            str(manifest),
            "--out",
            str(out_path),
            "--max_wer",
            str(args.max_wer),
            "--min_dur",
            str(args.min_dur),
            "--max_dur",
            str(args.max_dur),
        ]
        if args.batch_size:
            cmd.extend(["--batch_size", str(args.batch_size)])
        run(cmd, cwd=repo_root)
        return out_path

    cv_raw = out_dir / "cv17_ru.jsonl"
    ruls_raw = out_dir / "ruls.jsonl"
    pod_raw = out_dir / "podlodka.jsonl"
    golos_c_raw = out_dir / "golos_crowd.jsonl"
    golos_f_raw = out_dir / "golos_farfield.jsonl"

    bm("cv17-ru", cv_raw)
    bm("ruls", ruls_raw)
    bm("podlodka", pod_raw)
    bm("golos-crowd", golos_c_raw)
    bm("golos-farfield", golos_f_raw)

    cv_path = filt(cv_raw, out_dir / "cv17_ru_sel.jsonl")
    ruls_path = filt(ruls_raw, out_dir / "ruls_sel.jsonl")
    pod_path = filt(pod_raw, out_dir / "podlodka_sel.jsonl")
    golos_c_path = filt(golos_c_raw, out_dir / "golos_crowd_sel.jsonl")
    golos_f_path = filt(golos_f_raw, out_dir / "golos_farfield_sel.jsonl")

    nonspeech_path: Optional[Path] = None
    if args.include_nonspeech:
        nonspeech_path = out_dir / "nonspeech.jsonl"
        bm("audioset-nonspeech", nonspeech_path)

    telephony_path = filt(Path(args.telephony), out_dir / "telephony_sel.jsonl")

    golos_mix = out_dir / "golos_mix.jsonl"
    run(
        [
            py,
            "-m",
            "transcribe.tools.mix_manifests",
            "--in",
            f"crowd={golos_c_path}",
            "--in",
            f"farfield={golos_f_path}",
            "--out",
            str(golos_mix),
            "--concat",
        ],
        cwd=repo_root,
    )

    cmd = [
        py,
        "-m",
        "transcribe.tools.build_ru_stage1_mix",
        "--golos",
        str(golos_mix),
        "--cv",
        str(cv_path),
        "--ruls",
        str(ruls_path),
        "--podlodka",
        str(pod_path),
        "--telephony",
        str(telephony_path),
        "--out",
        str(out_dir / "ru_stage1_mix.jsonl"),
        "--target_size",
        str(args.target_size),
        "--shuffle",
        "--seed",
        str(args.seed),
        "--add_dataset_tag",
    ]
    if nonspeech_path:
        cmd.extend(["--nonspeech", str(nonspeech_path)])
    run(cmd, cwd=repo_root)


if __name__ == "__main__":
    main()
