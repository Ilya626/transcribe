#!/usr/bin/env python
"""
Runpod/Linux launcher for partial unfreezing fine-tuning of Canary (.nemo) with NeMo.

This script performs deeper fine-tuning than pure LoRA by selectively unfreezing
parts of the encoder/decoder (top K layers), optionally training norms/biases and
the output head. It reuses the same dataset pipeline as other scripts here (Lhotse).

Examples (A6000 48GB):

  # Unfreeze last 4 encoder layers + last 2 decoder layers + head, bf16
  python transcribe/training/runpod_nemo_canary_partial.py \
    --nemo /workspace/models/canary-1b-v2.nemo \
    --train /workspace/data/train_portable.jsonl \
    --val   /workspace/data/val_portable.jsonl \
    --outdir /workspace/exp/canary_partial_e4_d2 \
    --export /workspace/models/canary-partial-e4-d2.nemo \
    --unfreeze_encoder_last 4 --unfreeze_decoder_last 2 --unfreeze_head \
    --preset a6000-fast --early_stop --es_patience 4 --es_min_delta 0.003

Notes:
- By default, all params are frozen; requested subsets are unfrozen.
- You can still combine with LoRA ("hybrid") via --with_lora if desired.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
except Exception as e:  # pragma: no cover
    raise SystemExit(f"[ERR] Cannot import NeMo ASR models. Install nemo_toolkit. Underlying error: {e}")

# Make sure we can import local helpers
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:  # optional hybrid LoRA
    from transcribe.training.finetune_canary import (
        inject_lora_modules,
        merge_lora_into_linear,
        build_cuts_from_jsonl,
    )
except Exception:  # pragma: no cover
    inject_lora_modules = None  # type: ignore
    merge_lora_into_linear = None  # type: ignore
    def build_cuts_from_jsonl(path: str):  # type: ignore
        raise RuntimeError("build_cuts_from_jsonl unavailable; please run from repository root")


class CudaMemReport(Callback):
    def __init__(self, every_n_steps=50):
        super().__init__()
        self.every = int(max(1, every_n_steps))

    def _report(self, tag: str):
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        try:
            free, _ = torch.cuda.mem_get_info()
        except Exception:
            free = 0
        alloc = torch.cuda.memory_allocated(dev)
        reserv = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        gb = 1024 ** 3
        print(f"[MEM:{tag}] alloc={alloc/gb:.2f}G reserved={reserv/gb:.2f}G free={free/gb:.2f}G total={total/gb:.2f}G peak={peak/gb:.2f}G")

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


def _flag_present(name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in sys.argv[1:])


def apply_preset(args):
    if not getattr(args, "preset", None):
        return args
    preset = args.preset
    if preset == "a6000-fast":
        preset_vals = dict(bs=12, accum=1, num_workers=12)
    elif preset == "a6000-max":
        preset_vals = dict(bs=16, accum=1, num_workers=12)
    else:
        return args
    for k, v in preset_vals.items():
        flag = "--" + k.replace("_", "-")
        if not _flag_present(flag):
            setattr(args, k, v)
    return args


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_module(m):
    for p in m.parameters(recurse=True):
        p.requires_grad = True


def unfreeze_last_layers(model, attr_name: str, last_n: int) -> int:
    if last_n <= 0:
        return 0
    mod = getattr(model, attr_name, None)
    if mod is None:
        return 0
    # common names for layer containers
    for layers_attr in ("layers", "encoder_layers", "decoder_layers", "conformer_layers"):
        layers = getattr(mod, layers_attr, None)
        if layers is not None and hasattr(layers, "__len__"):
            n = len(layers)
            cnt = 0
            for i in range(max(0, n - last_n), n):
                try:
                    _unfreeze_module(layers[i])
                    cnt += sum(p.numel() for p in layers[i].parameters())
                except Exception:
                    pass
            return cnt
    # Fallback: regex over named_modules to pick highest indices
    names = []
    for name, m in mod.named_modules():
        if re.search(r"layer\D*(\d+)", name):
            names.append((name, m))
    if names:
        # sort by the numeric suffix if present
        def idx(nm):
            m = re.search(r"(\d+)", nm)
            return int(m.group(1)) if m else -1
        names.sort(key=lambda x: idx(x[0]))
        pick = names[-last_n:]
        cnt = 0
        for _, m in pick:
            _unfreeze_module(m)
            cnt += sum(p.numel() for p in m.parameters())
        return cnt
    return 0


def unfreeze_head(model) -> int:
    cnt = 0
    for name in ("log_softmax", "token_classifier", "ctc_decoder", "decoder", "proj", "head"):
        m = getattr(model, name, None)
        if m is not None:
            _unfreeze_module(m)
            cnt += sum(p.numel() for p in m.parameters())
    return cnt


def unfreeze_norms_and_bias(model, train_norms: bool, train_bias: bool) -> tuple[int, int]:
    n_cnt = b_cnt = 0
    if train_norms:
        for m in model.modules():
            cls = m.__class__.__name__.lower()
            if "layernorm" in cls or "batchnorm" in cls or "rmsnorm" in cls:
                for p in m.parameters(recurse=False):
                    if not p.requires_grad:
                        p.requires_grad = True
                        n_cnt += p.numel()
    if train_bias:
        for n, p in model.named_parameters():
            if n.endswith(".bias") and not p.requires_grad:
                p.requires_grad = True
                b_cnt += p.numel()
    return n_cnt, b_cnt


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--outdir", default="/workspace/exp/canary_partial")
    ap.add_argument("--export", default="/workspace/models/canary-partial.nemo")

    # Partial unfreeze controls
    ap.add_argument("--unfreeze_encoder_last", type=int, default=4)
    ap.add_argument("--unfreeze_decoder_last", type=int, default=2)
    ap.add_argument("--unfreeze_head", action="store_true")
    ap.add_argument("--train_norms", action="store_true")
    ap.add_argument("--train_bias", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true", help="Enable encoder gradient checkpointing if supported")

    # Optional hybrid LoRA
    ap.add_argument("--with_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Trainer presets & logging
    ap.add_argument("--preset", choices=["a6000-fast", "a6000-max"], default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--log", choices=["csv", "tb", "none"], default="csv")
    ap.add_argument("--val_every_n_steps", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--mem_report_steps", type=int, default=50)

    # Optimization & early stop
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--monitor", default="val_loss")
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--es_patience", type=int, default=3)
    ap.add_argument("--es_min_delta", type=float, default=0.003)
    ap.add_argument("--gradient_clip_val", type=float, default=1.0)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = apply_preset(args)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(Path(args.export).parent).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Restoring base model: {args.nemo}")
    model = EncDecMultiTaskModel.restore_from(
        restore_path=str(args.nemo),
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Freeze all and unfreeze requested subsets
    freeze_all(model)
    n_enc = unfreeze_last_layers(model, "encoder", int(args.unfreeze_encoder_last))
    n_dec = unfreeze_last_layers(model, "transf_decoder", int(args.unfreeze_decoder_last))
    n_head = unfreeze_head(model) if args.unfreeze_head else 0
    n_norm, n_bias = unfreeze_norms_and_bias(model, args.train_norms, args.train_bias)

    # Optional encoder grad checkpointing
    if args.grad_ckpt:
        try:
            enc = getattr(model, "encoder", None)
            if enc is not None and hasattr(enc, "gradient_checkpointing"):
                enc.gradient_checkpointing = True
                print("[CKPT] encoder.gradient_checkpointing=True")
        except Exception as e:
            print("[CKPT][WARN] cannot enable grad ckpt:", e)

    # Optional hybrid LoRA on top of partial unfreeze
    used_hybrid_lora = False
    if args.with_lora:
        if inject_lora_modules is None:
            print("[LoRA][WARN] inject_lora_modules not available; skipping hybrid LoRA")
        else:
            patterns = [
                r"(self_)?attn\.(q_proj|k_proj|v_proj|out_proj)",
                r"linear_qkv",
                r"linear_proj",
                r"linear_fc1",
                r"linear_fc2",
                r"\\bfc1\\b",
                r"\\bfc2\\b",
                r"linear1",
                r"linear2",
                r"\\bproj\\b",
                r"to_q\\b",
                r"to_k\\b",
                r"to_v\\b",
                r"to_out\\b",
            ]
            replaced, fb = inject_lora_modules(
                model,
                patterns,
                r=int(args.lora_r),
                alpha=int(args.lora_alpha),
                dropout=float(args.lora_dropout),
                fallback_prefixes=("encoder", "transf_decoder"),
                debug_print=0,
            )
            print(f"[LoRA] hybrid injected modules: {replaced} (prefix_fallback={fb})")
            used_hybrid_lora = replaced > 0

    # Best-effort optimizer overrides
    try:
        cfg = getattr(model, "cfg", None) or getattr(model, "_cfg", None)
        if cfg is not None and hasattr(cfg, "optim"):
            if args.lr is not None and hasattr(cfg.optim, "lr"):
                old = cfg.optim.lr
                cfg.optim.lr = args.lr
                print(f"[OPT] lr: {old} -> {cfg.optim.lr}")
            if args.weight_decay is not None and hasattr(cfg.optim, "weight_decay"):
                old = cfg.optim.weight_decay
                cfg.optim.weight_decay = args.weight_decay
                print(f"[OPT] weight_decay: {old} -> {cfg.optim.weight_decay}")
    except Exception as e:
        print("[OPT][WARN] cannot override optim cfg:", e)

    # Dataset via Lhotse cuts
    out_dir = Path("data"); out_dir.mkdir(exist_ok=True, parents=True)
    tr_cuts = out_dir / "train_cuts.jsonl.gz"
    va_cuts = out_dir / "val_cuts.jsonl.gz"
    if not tr_cuts.exists():
        cuts_tr, _ = build_cuts_from_jsonl(args.train)
        cuts_tr.to_file(tr_cuts)
    if not va_cuts.exists():
        cuts_va, _ = build_cuts_from_jsonl(args.val)
        cuts_va.to_file(va_cuts)

    train_cfg = {
        "use_lhotse": True,
        "cuts_path": str(tr_cuts),
        "num_workers": int(args.num_workers),
        "shuffle": True,
        "batch_size": int(args.bs),
        "pretokenize": False,
        "pin_memory": True,
    }
    val_cfg = {
        "use_lhotse": True,
        "cuts_path": str(va_cuts),
        "num_workers": int(args.num_workers),
        "shuffle": False,
        "batch_size": int(args.bs),
        "pretokenize": False,
        "pin_memory": True,
    }
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

    step_ckpt = ModelCheckpoint(
        dirpath=args.outdir,
        filename="step{step:06d}",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=int(args.save_every),
    )
    best_ckpt = ModelCheckpoint(
        dirpath=args.outdir,
        filename="best",
        save_top_k=1,
        monitor=args.monitor,
        mode="min",
    )
    mem_cb = CudaMemReport(every_n_steps=int(args.mem_report_steps))
    cbs = [step_ckpt, best_ckpt, mem_cb]
    if args.early_stop:
        cbs.append(EarlyStopping(monitor=args.monitor, mode="min", patience=int(args.es_patience), min_delta=float(args.es_min_delta)))
    if logger is not None:
        cbs.append(LearningRateMonitor(logging_interval="step"))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=int(args.max_steps),
        accumulate_grad_batches=int(args.accum),
        precision=args.precision,
        gradient_clip_val=float(args.gradient_clip_val),
        logger=logger,
        enable_checkpointing=True,
        callbacks=cbs,
        val_check_interval=int(args.val_every_n_steps),
        log_every_n_steps=10,
    )

    total_tr = count_trainable(model)
    print(
        f"[TRAIN] Partial FT | enc_last={args.unfreeze_encoder_last} dec_last={args.unfreeze_decoder_last} "
        f"head={bool(args.unfreeze_head)} norms={bool(args.train_norms)} bias={bool(args.train_bias)} "
        f"hybrid_lora={bool(args.with_lora)} | bs={args.bs} accum={args.accum} workers={args.num_workers} "
        f"precision={args.precision} | trainable_params={total_tr/1e6:.2f}M"
    )
    trainer.fit(model)

    print("[EXPORT] Saving .nemo")
    try:
        if args.with_lora and merge_lora_into_linear is not None:
            print("[EXPORT] Merging LoRA into base Linear ...")
            merge_lora_into_linear(model)
        model.save_to(str(args.export))
    except Exception as e:
        print("[ERR] Export failed:", e)
        sys.exit(1)
    print(f"[OK] Exported: {args.export}")


if __name__ == "__main__":
    main()

