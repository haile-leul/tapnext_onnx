# Windows / PowerShell version of gpu_quickstart.sh. Run from the repo root with the venv active:
#   .\scripts\gpu_quickstart.ps1 -Video C:\videos\gate.mp4 [-Q 64] [-TensorrtDir D:\path\to\tensorrt]
# Steps: reassemble ONNX -> check GPU env -> build TensorRT fp16 engine (+ verify)
#        -> benchmark ONNX Runtime CUDA vs TensorRT -> render tracks video + CSV.
param(
  [Parameter(Mandatory = $true)][string]$Video,
  [int]$Q = 64,
  [int]$BenchFrames = 200,
  [string]$TensorrtDir = $env:TENSORRT_DIR   # e.g. D:\SVN\PTnD\extern\tensorrt_10.14.1.48_CUDA13
)
$ErrorActionPreference = 'Stop'
if ($TensorrtDir) { $env:TENSORRT_DIR = $TensorrtDir }
function Run([string]$what, [scriptblock]$cmd) {
  Write-Host "`n=== $what ===" -ForegroundColor Cyan
  & $cmd
  if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

$Onnx   = 'models\tapnext_step_256_w16.onnx'
$Engine = "models\tapnext_step_256_fp16_q$Q.engine"
$Stem   = [IO.Path]::Combine([IO.Path]::GetDirectoryName((Resolve-Path $Video)),
                             [IO.Path]::GetFileNameWithoutExtension($Video))

Run 'reassemble model'      { python tools\reassemble_model.py }
Run 'check GPU environment' { python tools\check_gpu_env.py }
if (-not (Test-Path $Engine)) {
  Run 'build TensorRT engine' { python tools\build_trt_engine.py --onnx $Onnx --engine $Engine --min-q $Q --opt-q $Q --max-q $Q }
} else {
  Write-Host "`n$Engine exists, skipping build (delete it to rebuild)"
}
Run 'benchmark ONNX Runtime CUDA' { python run_video.py --video $Video --backend ort-cuda --model $Onnx --slots $Q --grid 8 --benchmark --max-frames $BenchFrames }
Run 'benchmark TensorRT'          { python run_video.py --video $Video --backend trt --model $Engine --slots $Q --grid 8 --benchmark --max-frames $BenchFrames }
Run 'render tracks (TensorRT)'    { python run_video.py --video $Video --backend trt --model $Engine --slots $Q --grid 8 --reseed-every 15 --reseed-grid 6 --drop-after 5 --out "${Stem}_tapnext_trt.mp4" --csv "${Stem}_tapnext_trt.csv" }
if ($env:TENSORRT_DIR -and (Test-Path "$env:TENSORRT_DIR\bin\trtexec.exe")) {
  $n = 1024 + $Q
  Run 'trtexec (your C++ TensorRT install)' { & "$env:TENSORRT_DIR\bin\trtexec.exe" --loadEngine=$Engine "--shapes=query_points:1x${Q}x3,point_reset:1x${Q},lru_state:12x${n}x768,conv_state:12x${n}x3x768" --useCudaGraph --noDataTransfers }
}
Write-Host "`nDone: ${Stem}_tapnext_trt.mp4" -ForegroundColor Green
