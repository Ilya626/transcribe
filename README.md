# Speech Model Inference and Evaluation

This repository contains scripts to run speech recognition inference with several open-source models and to evaluate their quality.

## Documentation

For focused guides, see the documentation in `docs/`:

- [Setup & Environment](docs/ENV.md)
- [Datasets & Manifests](docs/DATA.md)
- [Packing Dataset (portable paths)](docs/PACK_DATA.md)
- [Inference Runbook](docs/RUNBOOK.md)
- [Models Reference](docs/MODELS.md)
- [Evaluation & Analysis](docs/EVAL.md)
- [Fine-tuning](docs/FINETUNE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Whisper Variants](docs/WHISPER_VARIANTS.md)
- [Runpod Setup & Launch](docs/RUNPOD.md)

An index linking these guides is available at [docs/INDEX.md](docs/INDEX.md).

## Inference

Each model has its own script under `models/`:

- `inference_canary.py` - [NVIDIA Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2)
- `inference_whisper.py` - [OpenAI Whisper large v3](https://huggingface.co/openai/whisper-large-v3)
  Also supports other Whisper-compatible models via `--model_id`, e.g.
  [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  and [distil-whisper/distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5).
- `inference_gigaam.py` – [Salute GigaAM v2](https://github.com/salute-developers/GigaAM)
- `inference_silero.py` – [Silero STT](https://github.com/snakers4/silero-models)
- `inference_vosk.py` – [Vosk](https://alphacephei.com/vosk)

Usage is similar for all scripts:

```bash
python models/inference_whisper.py <path/to/audio_or_dir> predictions.json
```

Results are written as JSON mapping audio paths to transcribed text.

Whisper language/task (forces no autodetect/translate):

```
python transcribe/models/inference_whisper.py data/train.jsonl preds.json \
  --model_id distil-whisper/distil-large-v3.5 --language ru --task transcribe
```

Defaults for Whisper batch size and beams (12 GB GPU):
- openai/whisper-large-v3: batch_size=24, num_beams=1
- openai/whisper-large-v3-turbo: batch_size=64, num_beams=1
- distil-whisper/distil-large-v3.5: batch_size=64, num_beams=1
You can override with `--batch_size` and `--num_beams`.

For a broader overview of Russian speech recognition projects, see resources like the [Alpha Cephei blog](https://alphacephei.com/nsh/2025/04/18/russian-models.html) which surveys Vosk, Silero, Whisper and other models.

## Evaluation

To compare predictions with reference transcripts, use:

```bash
python evaluation/evaluate_transcriptions.py reference.json predictions.json
```

The script prints word error rate (WER), character error rate (CER), sentence error rate (SER) and a semantic similarity score. It
also reports counts of substitutions, insertions and deletions.

To evaluate several model outputs in one run:

```bash
python evaluation/compare_transcriptions.py reference.json whisper.json canary.json gigaam.json
```

This prints the same set of metrics for each predictions file, allowing quick comparison between models.

## Advanced Evaluation

For deeper error analysis across multiple models (alignment, taxonomy, confusions, optional NER and semantic similarity), use the advanced CLI:

```
python -m transcribe.evaluation.analyze_errors data/train.jsonl \
  --pred openai_v3=transcribe/preds/probe_large_v3_bs16.json \
  --pred turbo=transcribe/preds/probe_large_v3_turbo_bs32.json \
  --outdir transcribe/preds/analysis
```

Outputs under `--outdir` include:
- `per_utterance.csv`: per-utterance metrics and tags (WER/CER/SER, class, categories, confusions).
- `models.csv`: aggregated metrics per model.
- `models_with_baseline.csv`: aggregates plus relative deltas to a baseline pattern and cosine to baseline.
- `top_confusions.csv`: most frequent ref→hyp substitutions.
- `disagreement.csv`: disagreement by utterance (std of WER across models).
- `taxonomy.json`: shares of operations (sub/ins/del).
- `summary.html` and `summary.json`: compact, human-readable summary.

Options:
- `--no_semantic`: disable semantic similarity (faster, offline).
- `--no_ner`: disable named-entity extraction.
- `--st_model`: sentence-transformers model name (default `paraphrase-multilingual-MiniLM-L12-v2`).
- `--st_device`: device for sentence-transformers (`cpu` default; set `cuda` to use GPU).

Notes:
- References can be JSONL (`{"audio_filepath": ..., "text": ...}`) or JSON mapping `path->text`.
- Predictions are JSON mapping `path->hypothesis`; keys should match `audio_filepath` from references.
- NER uses `natasha` if available, otherwise a light capitalization heuristic for Russian.

Semantic setup (Windows):
- First run will download `sentence-transformers` model to `transcribe/.hf`. If you see a symlink warning, it is benign; to remove it, enable Windows Developer Mode or run as Admin.
- To prefetch the model explicitly:
  - `. transcribe/env.ps1; .\transcribe\.venv\Scripts\python transcribe/tmp/load_st_model.py`
 - Advanced CLI frees the semantic model at the end; if you run repeatedly in a long-lived process, call `release_semantic_models()` from `transcribe.evaluation.advanced.semantics` to free memory.

## Notes & Tips

- Recommended multi-model advanced evaluation (excluding Distil by default):

```
python -m transcribe.evaluation.analyze_errors data/train.jsonl \
  --pred v3=transcribe/preds/whisper_large_v3_train_beam2_bs8.json \
  --pred turbo=transcribe/preds/whisper_large_v3_turbo_train_beam2_bs10.json \
  --pred ru_antony66=transcribe/preds/ru_whisper_large_v3_antony66_train.json \
  --pred ru_apelsin=transcribe/preds/ru_whisper_large_v3_apelsin_train.json \
  --pred ru_dvislobokov=transcribe/preds/ru_whisper_large_v3_turbo_dvislobokov_train.json \
  --pred ru_bond005=transcribe/preds/ru_whisper_podlodka_turbo_bond005_train.json \
  --outdir transcribe/preds/analysis_whisper_full --st_device cuda
```

- Additional analyze options:
  - `--st_bs`: sentence-transformers batch size (default 256)
  - `--sample N --seed S`: analyze a random subset
- Evaluation additionally normalizes `ё`→`е` for robust matching.

- Whisper inference tips:
  - Always add `--language ru --task transcribe` to avoid auto-translation.
  - Optional `--engine pipeline` enables chunked long-form decoding (`chunk_length_s=30`).
  - The inference script prevents parallel runs via a simple GPU lock and explicitly cleans CUDA memory.

## Finetuning

Experimental fine-tuning of Whisper models is supported via
`training/finetune_whisper.py`.  It expects two JSON manifests with
fields `audio` (path to a WAV file) and `text` (reference transcript):

```bash
python training/finetune_whisper.py train.json eval.json out_dir \
    --model_id openai/whisper-small --batch_size 2 --num_epochs 1
```

The defaults aim at a Windows desktop with an RTX&nbsp;4070&nbsp;Ti (12&nbsp;GB).
Adjust the model size or batch parameters if you hit out-of-memory
errors.

For NVIDIA's Canary models, `training/finetune_canary.py` performs
LoRA-based fine-tuning using the NeMo toolkit and Lhotse manifests.  It
requires JSONL manifests with `audio_filepath` and `text` fields:

```bash
python training/finetune_canary.py --nemo path/to/model.nemo \
    --train train.jsonl --val val.jsonl
```

The script injects LoRA adapters, runs training and exports a merged
`.nemo` checkpoint on completion.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
# For GigaAM
pip install git+https://github.com/salute-developers/GigaAM
```

Ensure `ffmpeg` is available for audio decoding.

## Quick Start (GPU-only)

- Activate local env and caches in PowerShell:

```
. transcribe/env.ps1
.\transcribe\.venv\Scripts\Activate.ps1
```

- All model downloads and temp files are stored inside the repo under `.hf`, `.torch`, `.tmp`.
- Inference on CPU is disabled by design – CUDA GPU is required.

Text normalization in evaluation unifies `ё`→`е` for robust matching.

## Environment & Caches

This project stores all model caches and temporary files inside the repo to avoid using system disks. The PowerShell helper `transcribe/env.ps1` sets these variables for the current shell session and ensures directories exist:

- HF_HOME: base directory for Hugging Face caches. Default: `transcribe/.hf`.
- TRANSFORMERS_CACHE: additional cache path used by `transformers`. Default: `transcribe/.hf`.
- HF_HUB_CACHE: location for HF Hub model shards. Default: `transcribe/.hf/hub`.
- TORCH_HOME: Torch/Torch Hub cache directory. Default: `transcribe/.torch`.
- TMP, TEMP: temp directory used by Python and the scripts. Default: `transcribe/.tmp`.
- PIP_CACHE_DIR: pip download/cache directory. Default: `transcribe/.pip-cache`.

Notes:
- Variables are set only for the current PowerShell process. Re-run `. transcribe/env.ps1` in new shells.
- Set `CUDA_VISIBLE_DEVICES` if you want to restrict which GPU is used.
- Inference scripts use a simple GPU lock file at `transcribe/.tmp/gpu.lock` to prevent concurrent runs. If a stale lock remains after a crash, it can be removed once no process is using the GPU.

Python environment tips (Windows):
- Prefer Python 3.11/3.12 for better binary wheel availability (ONNX, SentencePiece). Example:
  - `py -3.12 -m venv transcribe\.venv`
  - `. transcribe/env.ps1; .\transcribe\.venv\Scripts\Activate.ps1`

Additional notes:
- Transformers deprecation: you may see a warning that `TRANSFORMERS_CACHE` is deprecated. We still set it for compatibility, but `HF_HOME` is the primary cache root.
- HF Hub on Windows can warn about symlinks; caches still work. Developer Mode removes the warning.
- Restrict GPU: set `CUDA_VISIBLE_DEVICES=0` (or another id) before running any script.
- GPU lock: if a previous run crashed and left `transcribe/.tmp/gpu.lock`, delete it only after verifying no GPU process is running.

## Documentation Index

For a streamlined, task-focused set of docs, see:
- `transcribe/docs/INDEX.md` — hub page
- `transcribe/docs/ENV.md` — setup, environment, caches, CUDA
- `transcribe/docs/DATA.md` — dataset manifests and paths
- `transcribe/docs/RUNBOOK.md` — exact commands for inference per model
- `transcribe/docs/MODELS.md` — model-specific notes and args
- `transcribe/docs/EVAL.md` — evaluation and advanced analysis
- `transcribe/docs/FINETUNE.md` — finetuning guides
- `transcribe/docs/TROUBLESHOOTING.md` — common issues and fixes

CUDA/Torch setup (Windows):
- Install CUDA builds of torch matching your local drivers, for example CUDA 12.4:
  - `pip uninstall -y torch torchaudio`
  - `pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124`
- Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

GigaAM + SentencePiece:
- On Python 3.13, `sentencepiece` may fail to build; prefer Python 3.11/3.12 for prebuilt wheels. If you see CMake errors, recreate the venv with 3.12 and reinstall GigaAM.

## Whisper Family

Run on a JSONL manifest (one object per line with `audio_filepath`, `text`):

```
python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/whisper_large_v3_train.json \
  --model_id openai/whisper-large-v3 --num_beams 1

python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/whisper_large_v3_turbo_train.json \
  --model_id openai/whisper-large-v3-turbo --num_beams 1

python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/distil_large_v35_train.json \
  --model_id distil-whisper/distil-large-v3.5 --num_beams 1

# Russian finetunes
python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/ru_antony66_v3.json \
  --model_id antony66/whisper-large-v3-russian
python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/ru_dvislobokov_turbo.json \
  --model_id dvislobokov/whisper-large-v3-turbo-russian
python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/ru_apelsin_v3.json \
  --model_id Apel-sin/whisper-large-v3-russian-ties-podlodka-v1.2
python transcribe/models/inference_whisper.py data/train.jsonl transcribe/preds/ru_bond005_turbo.json \
  --model_id bond005/whisper-podlodka-turbo
```

Defaults for batch size on 12 GB GPU (override with `--batch_size`):
- openai/whisper-large-v3: 24
- openai/whisper-large-v3-turbo: 64
- distil-whisper/distil-large-v3.5: 64
- RU finetunes of large‑v3 (e.g. antony66, Apel‑sin): 24
- RU finetunes of turbo (e.g. dvislobokov, bond005): 64

`--num_beams` default is 1 for speed. During inference VRAM usage is printed per batch.

## Canary

There are two ways to run Canary:

- Transformers route (may not work for all Canary releases on HF, since the official artifact is a `.nemo` file):

```
python transcribe/models/inference_canary.py data/train.jsonl transcribe/preds/canary_v2_train.json \
  --model_id nvidia/canary-1b-v2 --language ru --task transcribe --num_beams 1 --engine pipeline
```

- NeMo route (recommended; loads the official `.nemo` checkpoint):

```
python transcribe/models/inference_canary_nemo.py data/train.jsonl transcribe/preds/canary_v2_train.json \
  --batch_size 32 --source_lang ru --target_lang ru --task asr --pnc yes
```

Notes for NeMo route:
- Requires `nemo_toolkit`, `lhotse`, and common audio libs. If some imports are missing, install: `pip install -U nemo_toolkit lhotse librosa einops pandas jiwer editdistance pyannote.core pyannote.metrics webdataset`.
- Windows/ONNX + NeMo may require NumPy < 2 for some binary wheels: `pip install -U --only-binary=:all: numpy==1.26.4`.
- The script downloads `canary-1b-v2.nemo` into `transcribe/.hf/models--nvidia--canary-1b-v2`.
- Batch size guidance (12 GB GPU): 16–32 обычно работает стабильно; при OOM снизьте до 22/16. Скрипт освобождает CUDA память между батчами и печатает VRAM‑отчеты.

## GigaAM

After installing GigaAM extras (see Requirements), you can transcribe a JSONL/JSON manifest or a file/directory:

```
python transcribe/models/inference_gigaam.py data/train.jsonl transcribe/preds/gigaam_v2_rnnt_train.json \
  --model v2_rnnt --batch_size 16
```

Note: GigaAM API returns per‑file results; batching groups files for progress/VRAM reporting.

GigaAM setup tips (Windows):
- Prefer Python 3.11/3.12 (prebuilt `sentencepiece` wheels). If using 3.13 and you see CMake errors, recreate the venv with 3.12 and run `pip install gigaam`.
- Ensure CUDA torch is installed (see CUDA/Torch setup above) so `torch.cuda.is_available()` is True.

Long-form audio (fallback):
- If a file is too long, the script falls back from `model.transcribe()` to `model.transcribe_longform()` when available.
- `transcribe_longform` uses pyannote VAD and may require a Hugging Face token:
  - Set `HF_TOKEN` for the current shell: `Set-Item env:HF_TOKEN 'hf_xxx'` (PowerShell) or `$env:HF_TOKEN='hf_xxx'`.
  - Install dependencies: `pip install -U pyannote.audio`.
- Alternatively, prepare a short-audio manifest (e.g., ≤ 25 s) to avoid long-form entirely.

Short-audio manifest (Python one-liner):
```
python - << 'PY'
import json, soundfile as sf
src='data/train.jsonl'; out='transcribe/tmp/train_64_short25.jsonl'
n=0
with open(src,'r',encoding='utf-8') as fin, open(out,'w',encoding='utf-8') as fo:
  for ln in fin:
    if not ln.strip(): continue
    o=json.loads(ln); p=o.get('audio_filepath') or o.get('audio')
    if not p: continue
    try:
      info=sf.info(p); dur=info.frames/info.samplerate
    except Exception:
      continue
    if dur<=25.0:
      fo.write(ln); n+=1
      if n>=64: break
print('wrote', out, 'count', n)
PY
```

Memory & VRAM:
- The GigaAM script frees CUDA memory between batches and prints `[VRAM:batch_N]` and `[VRAM:final]` lines to help spot leaks.
- A simple GPU lock at `transcribe/.tmp/gpu.lock` prevents concurrent runs.

About “batch size” for GigaAM:
- The upstream API exposes `model.transcribe(path: str)` for a single file. There is no native GPU batching for multiple files.
- The script’s `--batch_size` only controls how many files are grouped per loop iteration (progress/VRAM reporting); it does not execute them in parallel on the GPU.
- Therefore, values like 32/64/128 are safe but won’t change per‑file GPU concurrency. For real concurrency you’d need a custom multi‑process pipeline with careful GPU coordination (not recommended by default).

Parallel instances (sharding):
- Если нужен параллельный запуск нескольких инстансов на одной GPU, используйте шардинг входного списка и отключите лок‑файл:
  - Добавлены флаги: `--no_lock`, `--shard_idx`, `--shard_total`.
  - Пример (6 процессов в PowerShell):
    ```powershell
    $N=6; 0..($N-1) | ForEach-Object {
      Start-Process -NoNewWindow -FilePath python -ArgumentList @(
        'transcribe/models/inference_gigaam.py',
        'transcribe/tmp/train_128_short30.jsonl',
        'transcribe/preds/gigaam_train_128_shard$_.json',
        '--batch_size', '64', '--no_lock', '--shard_idx', "$_", '--shard_total', "$N"
      )
    }
    ```
  - По завершении объедините JSON‑ы (ключи не конфликтуют, это `path -> text`).
  - Внимание: каждый процесс держит копию модели в VRAM (~0.5–1.0 GB в зависимости от версии и данных). На 12 GB GPU разумно 2–6 процессов, тестируйте по VRAM‑логам.

 Автопараллельный пуск (1 командой):
 - Упростили запуск — родительский процесс может сам запустить несколько шардов и слить результат:
   - Флаг `--parallel N` запускает N дочерних экземпляров с `--no_lock --shard_idx i --shard_total N`.
   - Пример (3 параллельных процесса):
     ```powershell
     python transcribe/models/inference_gigaam.py transcribe/tmp/train_128_short30.jsonl `
       transcribe/preds/gigaam_train_128_par3.json --batch_size 64 --parallel 3
     ```
   - Родитель ждёт завершения процессов, собирает `*_shard_i.json` и пишет итоговый JSON в файл из аргумента `output`.

Quick evaluation example:
```
# Transcribe 64 short utterances
python transcribe/models/inference_gigaam.py transcribe/tmp/train_64_short25.jsonl transcribe/preds/gigaam_train_64.json --batch_size 64

# Evaluate vs references
python transcribe/evaluation/evaluate_transcriptions.py data/train.jsonl transcribe/preds/gigaam_train_64.json --model paraphrase-multilingual-MiniLM-L12-v2
```

## Dataset Formats

- References (JSON): `{ "path/to/audio.wav": "transcript", ... }`
- References (JSONL): one object per line: `{ "audio_filepath": "...", "text": "..." }`
- Predictions (JSON): `{ "path/to/audio.wav": "hypothesis", ... }`

To convert JSONL → JSON references (PowerShell):

```
$in = 'data/train.jsonl'; $out = 'data/train_refs.json'
$m = @{}; Get-Content $in | ForEach-Object { $o = $_ | ConvertFrom-Json; $m[$o.audio_filepath] = $o.text }
$m | ConvertTo-Json -Depth 1 | Set-Content -Encoding UTF8 $out
```

## Troubleshooting

- `onnx` may fail to build on Windows Python 3.13 — prefer Python 3.11/3.12 if needed.
- `huggingface_hub` symlink warning on Windows is benign; caching still works.
- JSONL with BOM is handled automatically, but prefer plain UTF‑8 without BOM.
