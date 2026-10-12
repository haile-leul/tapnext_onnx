# Track-On2 → ONNX / TensorRT (Python + C++)

This repo deploys [Track-On2 / Track-On-R](https://github.com/gorkaydemir/track_on), an online point tracker with per-point memory. It covers:

- **Export:** two ONNX graphs.
- **TensorRT:** fp16 engines plus a check against ONNX Runtime.
- **Python runtime:** PyTorch (original code), ONNX Runtime (CPU/CUDA) and TensorRT backends.
- **Video tool:** `run_video.py` adds and removes points at any frame.
- **C++:** a header-only TensorRT tracker.

**No mmcv needed.** Track-On2's deformable attention is an mmcv CUDA op that can't be exported. `trackon_export/msda.py` re-implements it in pure PyTorch with the same parameters and maths (`grid_sample`). That makes the model exportable, and the original repo runs without mmcv, including on Windows.

## How it works

| graph | input → output | cost |
|---|---|---|
| `trackon_encoder` | frame `[1,3,384,512]` RGB 0..255 → features `f4,f8,f16,f32,f_fused` | ~140 GFLOP / frame (DINOv3 ViT-S+, ViT-Adapter, FPN) |
| `trackon_tracker` | features + per-point state (N rows) → points, visibility, next state | ~30 GFLOP + 0.44 GFLOP / point |

For comparison, TAPNext needs ~357 GFLOP per frame at 64 points. Track-On2 needs ~191.

### Per-point state and adding/removing points

Each point is one row in three arrays: `q_init [N,256]`, `memory [N,72,256]` and `memory_mask [N,72]`. This is the same layout as the official streaming `Predictor`, with its inference memory of 72 frames.

- **Add a point:** append a row, and on that frame pass `is_new=1` and the point's `new_xy`. The graph samples the point's query feature from the current frame.
- **Remove a point:** delete its row.
- **Limit on N:** the ONNX axis allows 1…8192 points (`export_onnx.py --max-points`). TensorRT allows whatever profile you build (`--max-points`, default 1024).

Points interact only through the per-frame query attention, exactly as in the original model.

### Tracker inputs and outputs

| tracker input | shape | |
|---|---|---|
| `f4,f8,f16,f32,f_fused` | `[1,12288,256]`, `[1,3072,256]`, `[1,768,256]`, `[1,192,256]`, `[1,12288,256]` | encoder outputs |
| `q_init`, `memory`, `memory_mask` | `[N,256]`, `[N,72,256]`, `[N,72]` | state (`mask` 1 = empty) |
| `new_xy`, `is_new` | `[N,2]`, `[N]` | new points (512×384 model pixels) |

| tracker output | shape | |
|---|---|---|
| `points` | `[N,2]` | (x, y) in 512×384 model pixels |
| `vis_logit` | `[N]` | visible if `sigmoid ≥ 0.8` |
| `unc_logit` | `[N]` | uncertainty logit |
| `q_init_out`, `memory_out`, `memory_mask_out` | as inputs | next frame's state |

The frame (or `--roi`) is resized to 512×384 without keeping the aspect ratio, like the official Predictor. Coordinates are scaled back to the original frame.

## What you need

1. **The official repo, for model code:**
   ```bash
   git clone https://github.com/gorkaydemir/track_on ../track_on
   ```
2. **A checkpoint.** [Track-On-R](https://huggingface.co/gorkaydemir/track_on_r/resolve/main/track_on_r.pt?download=true) is recommended for real-world video. [Track-On2](https://huggingface.co/gorkaydemir/track_on2/resolve/main/trackon2_dinov3_checkpoint.pt?download=true) is the other option.
3. **DINOv3 backbone access.** The weights are gated:
   - request access at https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m
   - then run `huggingface-cli login` once

   The weights download automatically on first export. Because of their license, the exported ONNX files and engines are not committed here.

## Windows (PowerShell, venv active, repo root)

### 1. Packages

Use CUDA 13 everywhere, so it matches your TensorRT 10.14 for CUDA 13:

```powershell
$env:TENSORRT_DIR = "D:\SVN\PTnD\extern\tensorrt_10.14.1.48_CUDA13"
$env:TENSORRT_VERSION = "10.14.1.48"
pip uninstall -y onnxruntime
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install (Get-ChildItem "$env:TENSORRT_DIR\python\tensorrt-*-cp311-none-win_amd64.whl").FullName
huggingface-cli login
python tools\check_env.py --trackon-root ..\track_on       # must end with ALL GOOD
```

### 2. Export ONNX

This runs on CPU and takes a few minutes. It writes `models\trackon_encoder.onnx` and `models\trackon_tracker.onnx`, then compares ONNX Runtime with the original PyTorch code on a clip where points are added mid-way. Expect `OK`.

```powershell
python export_onnx.py --trackon-root ..\track_on --ckpt ..\track_on_r.pt --video D:\data\gate.mp4
```

### 3. Build TensorRT engines

These must be built with the same TensorRT as your C++ project. The build then checks the engines against ONNX Runtime, adding and removing points along the way. It prints the position error, NaN counts and GPU timings, and must end with `OK`.

```powershell
python tools\build_trt_engines.py --max-points 1024 --video D:\data\gate.mp4
```

### 4. Run videos

```powershell
# TensorRT, 8x8 grid, live window + output video next to the input
python run_video.py --video D:\data\gate.mp4 --backend trt --grid 8 --show
# add/remove points mid-video, fresh grids, drop lost points
python run_video.py --video D:\data\gate.mp4 --backend trt --grid 6 --add "40:812,430;900,455" --add-grid "55:700,350,300,200:5" --remove "80:0,1" --remove-box "120:0,0,400,1080" --reseed-every 30 --drop-after 15
# other backends, same pipeline (for speed / quality comparison)
python run_video.py --video D:\data\gate.mp4 --backend ort-cuda --grid 8 --benchmark --max-frames 150
python run_video.py --video D:\data\gate.mp4 --backend torch --trackon-root ..\track_on --ckpt ..\track_on_r.pt --grid 8 --benchmark --max-frames 150
# click points yourself on frames 0 and 60
python run_video.py --video D:\data\gate.mp4 --backend trt --pick 0 --pick 60 --show
```

Steps 2–4 in one go:

```powershell
.\scripts\windows_quickstart.ps1 -Video D:\data\gate.mp4 -TrackonRoot ..\track_on -Ckpt ..\track_on_r.pt
```

**Point options.** All coordinates are in original video pixels. Point ids are numbered 0, 1, 2… in the order points are added, and are shown in the CSV.

| option | effect |
|---|---|
| `--grid N` | N×N grid on the first frame (inside `--grid-box` if given) |
| `--add "F:x,y;…"` | add points at frame F |
| `--add-grid "F:x,y,w,h:N"` | add an N×N grid in a box at frame F |
| `--remove "F:id,…"` | remove points by id at frame F |
| `--remove-box "F:x,y,w,h"` | remove all points inside a box at frame F |
| `--reseed-every K` | add a fresh grid every K frames |
| `--drop-after N` | remove points that are invisible for N frames in a row |
| `--max-points` | cap; the oldest points are removed first |
| `--pick F` | click points on frame F |
| `--points-json` | read the schedule from a file |

**Output.**
- The video is written next to the input as `<name>_trackon.mp4`. It is finalised at the end or on Ctrl+C.
- `--csv` writes one row per point per frame: `frame,id,x,y,vx,vy,vis_prob,visible`.
- The timing summary shows the whole step and, for TensorRT, the pure GPU time of the encoder and the tracker.

`python tools\gpu_bench.py --trackon-root ..\track_on --ckpt ..\track_on_r.pt` compares your GPU's raw matmul TFLOPS with the model's time. Use it if things are far slower than expected.

### 5. C++ (TensorRT)

`cpp/trackon_trt.hpp` is header-only and needs TensorRT 10, the CUDA runtime and OpenCV:

```cpp
TrackOnTrt tracker("models/trackon_encoder.engine", "models/trackon_tracker.engine");
int id = tracker.addPoint(x, y);           // original pixels; tracked from the next step()
tracker.removePoint(id);                   // any time
const auto& pts = tracker.step(frameBgr);  // {id, x, y, vx, vy, visProb, visible, lost} per point
float vx, vy;
tracker.medianVelocity(vehicleBox, vx, vy);  // motion prior for faces inside that vehicle
```

The per-point state stays on the GPU, in two buffers that swap each frame:
- A removal copies the kept rows into the other buffer on the GPU.
- An add appends rows and grows capacity ×2 when needed.

Each frame, only the 512×384 frame goes up to the GPU and the N points come back.

```powershell
cmake -S cpp -B build -DTENSORRT_ROOT="$env:TENSORRT_DIR" -DOpenCV_DIR=<path to OpenCVConfig.cmake>
cmake --build build --config Release
.\build\Release\trackon_demo.exe models\trackon_encoder.engine models\trackon_tracker.engine D:\data\gate.mp4 out.mp4
```

## TensorRT precision

`--precision mixed` is the default. It runs fp16, but keeps these in fp32:
- every LayerNorm and GroupNorm;
- the tracker's coordinate maths: argmax/top-k → pixel coordinates → sampling grids → output points. fp16 only resolves 0.25 px at x≈512.

The tracker graph is also written to be fp16-safe: L2-normalisation is overflow-free, and attention is scaled before Q·Kᵀ.

If the check reports NaN or large errors, there are two fallbacks:
- `--tracker-precision fp32`, which matters only if you track many points;
- `--precision fp32`.

## What was verified here

This development machine has no GPU, and HuggingFace is blocked by its network policy. So everything below used the exact architecture with random weights for DINOv3 and the tracker.

- **Wrappers vs original code:** on 4 frames with points added at frame 2, compared with `extract_frame_features` + `track_frame` + the Predictor's memory update. Max difference 7.6e-6 px; encoder features identical.
- **ONNX Runtime vs original PyTorch:** max difference 3e-5 px. N changed 3 → 5 → 6, and N = 1, 2 and 500 also ran.
- **Removing points:** PyTorch and ONNX backends with points added and removed. Identical tracks (`run_video.py` CSVs match exactly).
- **TensorRT:** not run here (no GPU). The engine builder's tagging was checked on the exported graphs. The C++ tracker compiles cleanly (`-Wall -Wextra`) against the TensorRT 10.13 headers. `build_trt_engines.py` verifies the engines against ONNX Runtime on your machine.

## Files

| path | |
|---|---|
| `trackon_export/msda.py` | mmcv-free `MultiScaleDeformableAttention` + import shim |
| `trackon_export/loading.py` | build the model from the official repo + checkpoint (memory 24 → 72 like Predictor) |
| `trackon_export/modules.py` | export wrappers: `TrackOnEncoder`, `TrackOnTracker` |
| `export_onnx.py` | export both graphs + check against PyTorch |
| `tools/build_trt_engines.py` | TensorRT engines + check against ONNX Runtime |
| `trackon_runtime.py` | Python backends (`torch`, `ort-cpu`, `ort-cuda`, `trt`) + `PointTracker` (ids, add/remove, velocities) |
| `run_video.py` | video runner / renderer / benchmark |
| `tools/check_env.py`, `tools/gpu_bench.py` | environment check; GPU vs model speed |
| `cpp/trackon_trt.hpp`, `cpp/demo.cpp`, `cpp/CMakeLists.txt` | C++ TensorRT tracker + demo |
| `scripts/windows_quickstart.ps1` | steps 2–4 in one go |

Track-On is licensed by its authors (see the official repo). The DINOv3 weights are under Meta's DINOv3 license.
