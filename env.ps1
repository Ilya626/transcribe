$ErrorActionPreference = 'Stop'

# Project-local caches and temp directories (avoid system disk usage)
# Variables set for the current PowerShell process only:
# - PIP_CACHE_DIR       : pip's download/cache directory (transcribe/.pip-cache)
# - HF_HOME             : base Hugging Face cache dir (transcribe/.hf)
# - TRANSFORMERS_CACHE  : transformers cache dir (same as HF_HOME)
# - HF_HUB_CACHE        : HF Hub model shards (transcribe/.hf/hub)
# - TORCH_HOME          : PyTorch/Torch Hub cache dir (transcribe/.torch)
# - TMP, TEMP           : temporary directory (transcribe/.tmp)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$paths = @{
  PIP_CACHE_DIR        = Join-Path $projectRoot '.pip-cache'
  HF_HOME              = Join-Path $projectRoot '.hf'
  TRANSFORMERS_CACHE   = Join-Path $projectRoot '.hf'
  HF_HUB_CACHE         = Join-Path $projectRoot '.hf\hub'
  TORCH_HOME           = Join-Path $projectRoot '.torch'
  TMP                  = Join-Path $projectRoot '.tmp'
  TEMP                 = Join-Path $projectRoot '.tmp'
}

foreach ($kv in $paths.GetEnumerator()) {
  $dir = $kv.Value
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  [System.Environment]::SetEnvironmentVariable($kv.Key, $dir, 'Process')
  Set-Item -Path env:$($kv.Key) -Value $dir
}

Write-Output 'Project-local environment configured:'
($paths.GetEnumerator() | Sort-Object Name | ForEach-Object { "  $($_.Key) = $($_.Value)" })

Write-Output 'Tip: activate venv with:  .\transcribe\.venv\Scripts\Activate.ps1'
