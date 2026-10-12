#!/usr/bin/env bash
# Run from the repo root on a machine with an NVIDIA GPU:
#   bash scripts/gpu_quickstart.sh path/to/video.mp4 [num_slots]
# Steps: reassemble ONNX -> install deps -> build TensorRT fp16 engine (+ verify)
#        -> run the video with ONNX Runtime CUDA and with TensorRT.
set -euo pipefail
VIDEO=${1:?usage: bash scripts/gpu_quickstart.sh video.mp4 [num_slots]}
Q=${2:-64}
ONNX=models/tapnext_step_256_w16.onnx
ENGINE=models/tapnext_step_256_fp16_q${Q}.engine

# 1) reassemble + checksum
if [ ! -f "$ONNX" ]; then cat models/tapnext_step_256_w16.onnx.part* > "$ONNX"; fi
(cd models && sha256sum -c tapnext_step_256_w16.onnx.sha256)

# 2) Python deps (use a venv/conda env; pick the onnxruntime-gpu / tensorrt builds matching your CUDA)
python -m pip install -U numpy onnx opencv-python onnxruntime-gpu tensorrt torch

# 3) build the TensorRT engine for exactly Q slots (fastest) and check it against ONNX Runtime
python tools/build_trt_engine.py --onnx "$ONNX" --engine "$ENGINE" \
    --min-q "$Q" --opt-q "$Q" --max-q "$Q"

# 4) speed: ONNX Runtime CUDA vs TensorRT (no drawing)
python run_video.py --video "$VIDEO" --backend ort-cuda --model "$ONNX" --slots "$Q" --grid 8 --benchmark --max-frames 200
python run_video.py --video "$VIDEO" --backend trt --model "$ENGINE" --slots "$Q" --grid 8 --benchmark --max-frames 200

# 5) quality: rendered video + CSV (8x8 grid at frame 0, fresh 6x6 grid every 15 frames)
python run_video.py --video "$VIDEO" --backend trt --model "$ENGINE" --slots "$Q" \
    --grid 8 --reseed-every 15 --reseed-grid 6 --drop-after 5 \
    --out "${VIDEO%.*}_tapnext_trt.mp4" --csv "${VIDEO%.*}_tapnext_trt.csv"
echo "done: ${VIDEO%.*}_tapnext_trt.mp4"
