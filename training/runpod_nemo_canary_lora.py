#!/usr/bin/env python
"""
Runpod/Linux launcher for native NeMo LoRA fine-tuning of Canary.

Goals (per guide):
- Use NeMo native adapter (LoRA) instead of custom layer wrapping.
- Keep changes minimal and explicit; rely on NeMo APIs.

Requirements on the machine (e.g., Runpod A6000 48GB):
- Python 3.11/3.12 recommended
- torch / torchaudio CUDA builds (e.g., torch 2.6 + cu12x)
- nemo_toolkit, lhotse, lightning

Example:
  python transcribe/training/runpod_nemo_canary_lora.py \
    --nemo /workspace/models/canary-1b-v2.nemo \
    --train /workspace/data/train.jsonl \
    --val /workspace/data/val.jsonl \
    --outdir /workspace/exp/canary_ru_lora_a6000 \
    --export /workspace/models/canary-ru-lora-a6000.nemo \
    --bs 8 --accum 1 --precision bf16 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05

Notes:
- If your NeMo build lacks native adapter support, this script will raise with setup tips.
"""

import argparse
from pathlib import Path
import sys
import os
import shutil

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
except Exception as e:  # pragma: no cover
    raise SystemExit(f"[ERR] Cannot import NeMo ASR models. Install nemo_toolkit. Underlying error: {e}")


class CudaMemReport(Callback):
    def __init__(self, every_n_steps=50):
        super().__init__()
        self.every = int(max(1, every_n_steps))

    def _report(self, tag):
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        try:
            free, total_rt = torch.cuda.mem_get_info()
        except Exception:
            free, total_rt = 0, 0
        alloc = torch.cuda.memory_allocated(dev)
        reserv = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        gb = 1024 ** 3
        print(
            f"[MEM:{tag}] alloc={alloc/gb:.2f}G reserved={reserv/gb:.2f}G free={free/gb:.2f}G total={total/gb:.2f}G peak={peak/gb:.2f}G"
        )

    def on_fit_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._report("fit_start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step % self.every) == 0:
            self._report(f"step{trainer.global_step}")

    def on_validation_end(self, trainer, pl_module):
        self._report("val_end")

    def on_fit_end(self, trainer, pl_module):
        self._report("fit_end")


def ensure_nemo_adapter_api(model):
    """Check that NeMo adapter API appears to be available and return utility.

    We expect methods like `add_adapter` and possibly `set_enabled_adapters` on the model.
    Some NeMo versions use adapter configs with a Hydra-style dict (`_target_` pointing to LoRA).
    """
    if not hasattr(model, "add_adapter"):
        raise RuntimeError(
            "This NeMo build does not expose add_adapter on the model.\n"
            "Upgrade nemo_toolkit to a version with native adapters (LoRA) "
            "or use the Windows-friendly script finetune_canary.py (custom LoRA)."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nemo", required=False, default="/workspace/models/canary-1b-v2.nemo", help="Path to base .nemo checkpoint (e.g., canary-1b-v2.nemo)")
    ap.add_argument("--auto_download", action="store_true", help="Auto-download .nemo if missing via Hugging Face")
    ap.add_argument("--model_id", default="nvidia/canary-1b-v2", help="HF repo id for Canary .nemo")
    ap.add_argument("--nemo_name", default="canary-1b-v2.nemo", help="Filename of .nemo artifact in the repo")
    ap.add_argument("--train", required=True, help="Train JSONL manifest (audio_filepath/text/source_lang/target_lang/pnc)")
    ap.add_argument("--val", required=True, help="Val JSONL manifest")
    ap.add_argument("--outdir", default="/workspace/exp/canary_ru_lora_a6000")
    ap.add_argument("--export", default="/workspace/models/canary-ru-lora-a6000.nemo")
    ap.add_argument("--adapter_name", default="ru_lora")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--val_every_n_steps", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--mem_report_steps", type=int, default=50)
    ap.add_argument("--log", choices=["csv", "tb", "none"], default="csv")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(Path(args.export).parent).mkdir(parents=True, exist_ok=True)

    # Optionally download .nemo from HF if not present
    nemo_path = Path(args.nemo)
    if args.auto_download or not nemo_path.exists():
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise SystemExit("[ERR] huggingface_hub is required for auto-download. Install it or provide --nemo path.\n" + str(e))

        # Configure local caches if not set
        repo_root = Path(__file__).resolve().parents[2]
        os.environ.setdefault("HF_HOME", str(repo_root / ".hf"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])  # compat
        os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
        os.environ.setdefault("TORCH_HOME", str(repo_root / ".torch"))
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

        print(f"[DL] Downloading {args.model_id}:{args.nemo_name} from Hugging Face...")
        try:
            downloaded = hf_hub_download(repo_id=args.model_id, filename=args.nemo_name, token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            raise SystemExit(f"[ERR] Failed to download {args.model_id}/{args.nemo_name}: {e}")

        # If a destination path was requested, copy there; else use cache path
        if nemo_path.suffix.lower() == ".nemo":
            nemo_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(downloaded, nemo_path)
                print(f"[DL] Copied .nemo -> {nemo_path}")
            except Exception as e:
                print(f"[WARN] Could not copy to {nemo_path}: {e}. Using cache path instead: {downloaded}")
                nemo_path = Path(downloaded)
        else:
            nemo_path = Path(downloaded)

    print(f"[INFO] Restoring base model: {nemo_path}")
    model = EncDecMultiTaskModel.restore_from(
        restore_path=str(nemo_path),
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Ensure adapter API
    ensure_nemo_adapter_api(model)

    # Build a Hydra-style adapter config for NeMo LoRA
    # Reference target path may vary by NeMo version; common location:
    #   nemo.core.adapters.lora.LoRAConfig or nemo.core.adapters.LoRAConfig
    # We pass a dict to avoid tight coupling with specific class imports.
    lora_cfg = {
        "_target_": "nemo.core.adapters.LoRAConfig",
        "r": int(args.lora_r),
        "alpha": int(args.lora_alpha),
        "dropout": float(args.lora_dropout),
        # Apply to common linear/attention projections (best-effort; NeMo resolves internally)
        "enabled_modules": [
            "encoder.*",
            "transf_decoder.*",
        ],
    }

    print(f"[ADAPTER] Adding LoRA adapter '{args.adapter_name}' with cfg: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    try:
        model.add_adapter(name=args.adapter_name, cfg=lora_cfg)
        # Enable the adapter; method name can differ by version.
        if hasattr(model, "set_enabled_adapters"):
            model.set_enabled_adapters([args.adapter_name], enabled=True)
        elif hasattr(model, "enable_adapters"):
            model.enable_adapters(adapters=[args.adapter_name])
        else:
            print("[WARN] Could not find an explicit enable_adapters API; proceeding (adapter likely active by default)")
    except Exception as e:
        raise SystemExit(
            "[ERR] Failed to add/enable LoRA adapter via NeMo native API.\n"
            f"Underlying error: {e}\n"
            "Please ensure nemo_toolkit >= version with native LoRA adapters."
        )

    # Freeze base weights if adapter requires it (common practice)
    try:
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "adapter_parameters"):
            for p in model.adapter_parameters():
                p.requires_grad = True
    except Exception:
        pass

    # Data config via Lhotse cuts produced on-the-fly by model's setup methods
    train_cfg = {
        "use_lhotse": True,
        "cuts_path": str(Path("data") / "train_cuts.jsonl.gz"),  # will be written if not exists
        "num_workers": 8,  # Linux; Windows can use 0
        "bucketing_sampler": False,
        "shuffle": True,
        "batch_size": int(args.bs),
        "pretokenize": False,
        "input_manifest": args.train,
    }
    val_cfg = {
        "use_lhotse": True,
        "cuts_path": str(Path("data") / "val_cuts.jsonl.gz"),
        "num_workers": 8,
        "bucketing_sampler": False,
        "shuffle": False,
        "batch_size": int(args.bs),
        "pretokenize": False,
        "input_manifest": args.val,
    }

    # The model's setup_* functions in our Windows script generate Lhotse cuts from JSONL.
    # Here we mimic the same behavior by calling them with extended config that includes input_manifest.
    def _ensure_cuts(path_jsonl, out_path):
        try:
            from transcribe.training.finetune_canary import build_cuts_from_jsonl  # reuse helper
            cuts, _ = build_cuts_from_jsonl(path_jsonl)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            cuts.to_file(out_path)
        except Exception as e:
            print(f"[WARN] Could not prebuild cuts via helper: {e}. Proceeding; model may build internally if supported.")

    _ensure_cuts(args.train, train_cfg["cuts_path"])
    _ensure_cuts(args.val, val_cfg["cuts_path"])

    model.setup_training_data(train_cfg)
    model.setup_validation_data(val_cfg)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger = None
    if args.log == "csv":
        logger = CSVLogger(save_dir=args.outdir, name="pl_logs")
    elif args.log == "tb":
        logger = TensorBoardLogger(save_dir=args.outdir, name="tb_logs")

    ckpt_cb = ModelCheckpoint(
        dirpath=args.outdir,
        filename="step{step:06d}",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=int(args.save_every),
    )
    mem_cb = CudaMemReport(every_n_steps=int(args.mem_report_steps))
    callbacks = [ckpt_cb, mem_cb]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=int(args.max_steps),
        accumulate_grad_batches=int(args.accum),
        precision=args.precision,
        logger=logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        val_check_interval=int(args.val_every_n_steps),
        log_every_n_steps=10,
    )

    print("[TRAIN] Starting fit() with native NeMo adapter (LoRA)")
    trainer.fit(model)

    print("[EXPORT] Saving merged adapter model to .nemo")
    try:
        model.save_to(str(args.export))
    except Exception as e:
        print("[ERR] Export failed:", e)
        sys.exit(1)
    print(f"[OK] Exported: {args.export}")


if __name__ == "__main__":
    main()
