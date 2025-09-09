# GigaAM Guide
[← Back to Documentation Index](README.md)

After installing GigaAM extras (see Requirements), you can transcribe a JSONL/JSON manifest or a file/directory:

```
python transcribe/models/inference_gigaam.py data/train.jsonl transcribe/preds/gigaam_v2_rnnt_train.json \
  --model v2_rnnt --batch_size 16
```

Note: GigaAM API returns per‑file results; batching groups files for progress/VRAM reporting.

## Setup tips (Windows)
- Prefer Python 3.11/3.12 (prebuilt `sentencepiece` wheels). If using 3.13 and you see CMake errors, recreate the venv with 3.12 and run `pip install gigaam`.
- Ensure CUDA torch is installed (see CUDA/Torch setup) so `torch.cuda.is_available()` is True.

## Long-form audio fallback
- If a file is too long, the script falls back from `model.transcribe()` to `model.transcribe_longform()` when available.
- `transcribe_longform` uses pyannote VAD and may require a Hugging Face token:
  - Set `HF_TOKEN` for the current shell: `Set-Item env:HF_TOKEN 'hf_xxx'` (PowerShell) or `$env:HF_TOKEN='hf_xxx'`.
  - Install dependencies: `pip install -U pyannote.audio`.
- Alternatively, prepare a short-audio manifest (e.g., ≤ 25 s) to avoid long-form entirely.

### Short-audio manifest helper
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

## Memory & VRAM
- The GigaAM script frees CUDA memory between batches and prints `[VRAM:batch_N]` and `[VRAM:final]` lines to help spot leaks.
- A simple GPU lock at `transcribe/.tmp/gpu.lock` prevents concurrent runs.

## About “batch size”
- The upstream API exposes `model.transcribe(path: str)` for a single file. There is no native GPU batching for multiple files.
- The script’s `--batch_size` only controls how many files are grouped per loop iteration (progress/VRAM reporting); it does not execute them in parallel on the GPU.
- Therefore, values like 32/64/128 are safe but won’t change per‑file GPU concurrency. For real concurrency you’d need a custom multi‑process pipeline with careful GPU coordination (not recommended by default).

## Parallel instances (sharding)
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

### Автопараллельный пуск (1 командой)
- Упростили запуск — родительский процесс может сам запустить несколько шардов и слить результат:
  - Флаг `--parallel N` запускает N дочерних экземпляров с `--no_lock --shard_idx i --shard_total N`.
  - Пример (3 параллельных процесса):
    ```powershell
    python transcribe/models/inference_gigaam.py transcribe/tmp/train_128_short30.jsonl `
      transcribe/preds/gigaam_train_128_par3.json --batch_size 64 --parallel 3
    ```
  - Родитель ждёт завершения процессов, собирает `*_shard_i.json` и пишет итоговый JSON в файл из аргумента `output`.

## Quick evaluation example
```
# Transcribe 64 short utterances
python transcribe/models/inference_gigaam.py transcribe/tmp/train_64_short25.jsonl transcribe/preds/gigaam_train_64.json --batch_size 64

# Evaluate vs references
python transcribe/evaluation/evaluate_transcriptions.py data/train.jsonl transcribe/preds/gigaam_train_64.json --model paraphrase-multilingual-MiniLM-L12-v2
```

Back to [Inference Runbook](RUNBOOK.md)
