# Canary LoRA Fine‑Tuning — Tasks and Runbook

This checklist captures work for (1) careful verification/optimization of the current Windows local fine‑tune path and (2) preparing a Runpod/Linux script (PyTorch 2.6, A6000 48 GB) using more aggressive settings.

Related script: `transcribe/training/finetune_canary.py`

## 1) Windows local fine‑tuning (be very careful)

Constraints and invariants:
- Keep the current approach: no Megatron; LoRA is injected directly into `nn.Linear` layers via patterns (Windows compatibility).
- Do not introduce dependencies that fail on Windows (keep current imports and Lightning usage).
- Preserve current CLIs and defaults so existing runs don’t break.

Tasks
- Validation safety net:
  - Run a short dry‑run (small train subset) with `--debug_linear 20` and assert `replaced > 0`.
  - Capture `[MEM:fit_start]`, `[MEM:stepN]`, `[MEM:val_end]` and `[MEM:fit_end]` prints for a 5–10 minute trial to record VRAM envelope on RTX 4070 Ti 12 GB.
  - Verify `.nemo` export after Ctrl+C path (interrupt flow) and after full flow (merged LoRA via `merge_lora_into_linear`). Re‑load exported `.nemo` with `EncDecMultiTaskModel.restore_from()`.
- Precision & stability:
  - Confirm `--precision bf16` works (4070 Ti typically supports bfloat16). If not, fall back to `fp16` and document.
  - Keep `torch.set_float32_matmul_precision("high")` as is (ok on Windows).
- Batch/accum tuning for 12 GB:
  - Start: `--bs 2 --accum 8 --precision bf16` (baseline). Measure step time and peak VRAM.
  - Try `--bs 3 --accum 6` and `--bs 4 --accum 4` if headroom exists. Stop on OOM or throttling.
- LoRA hyperparams sweep (short runs):
  - `--lora_r {8,16}` × `--lora_alpha {16,32}` × `--lora_dropout {0.0,0.05}`. Keep best trade‑off (val loss / speed / stability).
- Data pipeline:
  - Confirm Lhotse cut generation paths are under `data/*.jsonl.gz` and re‑used between runs to avoid rebuild costs.
  - Keep `num_workers=0` on Windows to avoid DataLoader/process issues.
- Checkpointing & resume:
  - Verify `--save_every` works and `last.ckpt` reload via Lightning `Trainer(resume_from_checkpoint=...)` (add a README note if using manual resume).
- Documentation:
  - Record the best Windows settings (BS/ACCUM/precision/LoRA) and expected VRAM/throughput in docs.

Acceptance criteria
- Short dry‑run completes; LoRA modules attached; `.nemo` export validated (interrupt + full).
- Recommended stable Windows config documented with VRAM and step‑time bounds.

## 2) Runpod/Linux (PyTorch 2.6, A6000 48 GB)

Assumptions
- Linux image with CUDA 12.4 and PyTorch 2.6 (or official PyTorch CUDA image). GPU: A6000 (48 GB, BF16 capable).
- We can keep the same LoRA injection path (simple and robust), or later migrate to native NeMo adapters on Linux if needed.

Deliverables
- A bash launcher that:
  - Verifies Python/torch versions; installs `nemo_toolkit`, `lhotse`, `lightning` if missing.
  - Runs `finetune_canary.py` with more aggressive defaults for 48 GB.

Recommended aggressive defaults (starting point)
- Precision: `bf16`
- Batch size / accumulation: `--bs 8 --accum 1` (scale to 12/1 if headroom; confirm by VRAM reports)
- LoRA: `--lora_r 16 --lora_alpha 32 --lora_dropout 0.05`
- Logging: `--log csv`, `--mem_report_steps 50`

Command example
```
./transcribe/training/runpod_finetune_canary.sh \
  --nemo /workspace/models/canary-1b-v2.nemo \
  --train /workspace/data/train.jsonl \
  --val /workspace/data/val.jsonl \
  --outdir /workspace/exp/canary_ru_lora_a6000 \
  --export /workspace/models/canary-ru-lora-a6000.nemo
```

Next steps (optional Linux‑only improvements)
- Explore NeMo native adapter APIs (if stable) to replace pattern‑based injection, keeping parity with current training loop.
- Increase DataLoader workers (`num_workers=8`) — would require adding a CLI in `finetune_canary.py` (Windows can remain `0`).
- Consider enabling `torch.compile` (Linux) for minor speedup after verifying stability.

Acceptance criteria
- Script bootstraps environment on fresh Runpod, prints torch/CUDA info, and starts training with A6000 48 GB.
- VRAM telemetry confirms target batch settings are feasible; `.nemo` export loads back successfully.
