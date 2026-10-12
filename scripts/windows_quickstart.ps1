# Track-On2 end to end on Windows (PowerShell, venv active, run from the repo root):
#   .\scripts\windows_quickstart.ps1 -Video D:\data\gate.mp4 `
#       -TrackonRoot ..\track_on -Ckpt ..\track_on_r.pt `
#       -TensorrtDir D:\SVN\PTnD\extern\tensorrt_10.14.1.48_CUDA13
# Steps: environment check -> ONNX export (+verify) -> TensorRT engines (+verify)
#        -> speed: ONNX Runtime CUDA vs TensorRT -> rendered video with tracks.
param(
  [Parameter(Mandatory = $true)][string]$Video,
  [string]$TrackonRoot = '..\track_on',
  [Parameter(Mandatory = $true)][string]$Ckpt,
  [string]$TensorrtDir = $env:TENSORRT_DIR,
  [int]$MaxPoints = 1024,
  [int]$Grid = 8,
  [int]$BenchFrames = 150
)
$ErrorActionPreference = 'Stop'
if ($TensorrtDir) { $env:TENSORRT_DIR = $TensorrtDir }
function Run([string]$what, [scriptblock]$cmd) {
  Write-Host "`n=== $what ===" -ForegroundColor Cyan
  & $cmd
  if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

Run 'environment check' { python tools\check_env.py --trackon-root $TrackonRoot }
if (-not (Test-Path models\trackon_tracker.onnx)) {
  Run 'export ONNX (+ check against PyTorch)' { python export_onnx.py --trackon-root $TrackonRoot --ckpt $Ckpt --video $Video }
}
if (-not (Test-Path models\trackon_tracker.engine)) {
  Run 'build TensorRT engines (+ check against ONNX Runtime)' { python tools\build_trt_engines.py --max-points $MaxPoints --video $Video }
}
Run 'speed: ONNX Runtime CUDA' { python run_video.py --video $Video --backend ort-cuda --grid $Grid --benchmark --max-frames $BenchFrames }
Run 'speed: TensorRT'          { python run_video.py --video $Video --backend trt --grid $Grid --benchmark --max-frames $BenchFrames }
Run 'render tracks (TensorRT)' { python run_video.py --video $Video --backend trt --grid $Grid --reseed-every 30 --reseed-grid 6 --drop-after 15 --max-points $MaxPoints --csv "$([IO.Path]::ChangeExtension($Video, $null))_trackon.csv" }
Write-Host "`nDone. Output video next to the input: *_trackon.mp4" -ForegroundColor Green
