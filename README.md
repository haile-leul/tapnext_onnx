# TAPNext (BootsTAPNext, 256×256) → ONNX

This is a streaming ONNX export of [BootsTAPNext](https://github.com/google-deepmind/tapnet), the online point tracker from google-deepmind/tapnet. It uses the official `bootstapnext_ckpt.npz` checkpoint with the TRecViT-B/8 backbone at 256×256. It also includes C++ wrappers for ONNX Runtime and TensorRT that return per-point positions and velocities, a TensorRT fp16 engine builder, and a Python script that runs a video on the GPU and renders the tracks.

## Files

| file | what |
|---|---|
| `models/tapnext_step_256_w16.onnx.part00..04` (390 MB total, split for GitHub's 100 MB limit) | Weights are stored as fp16 and cast to fp32 inside the graph. All compute and I/O run in fp32. Max deviation from fp32 is 0.016 px. |
| `export_tapnext_onnx.py` | Rebuilds the full fp32 model (`tapnext_step_256.onnx`, 779 MB) from the checkpoint. |
| `tools/convert_fp16.py` | Converts the fp32 model into the `w16` file. |
| `tools/check_parity_torch.py`, `tools/check_onnx.py`, `tools/check_midstream.py` | Checks against the original PyTorch TAPNext. |
| `scripts/gpu_quickstart.sh`, `scripts/gpu_quickstart.ps1` | Run every GPU step in order (Linux / Windows). |
| `tools/reassemble_model.py`, `tools/check_gpu_env.py` | Join the model parts with a checksum check, and check the GPU environment. |
| `tools/build_trt_engine.py` | Builds the TensorRT fp16 engine on your GPU, then checks it against ONNX Runtime. |
| `run_video.py`, `tapnext_runtime.py` | Runs a video through ONNX Runtime (CPU/CUDA/TensorRT EP) or a TensorRT engine, with mid-video point adds, and writes a video with tracks, a CSV and timing. |
| `cpp/tapnext_points.hpp` | Shared C++ slot, ROI and velocity logic. |
| `cpp/tapnext_tracker.hpp` | C++ ONNX Runtime backend. |
| `cpp/tapnext_trt_tracker.hpp` | C++ TensorRT backend; the state stays on the GPU. |
| `cpp/demo.cpp`, `cpp/CMakeLists.txt` | C++ demo for both backends. |

## Reassembling the model

```bash
cat models/tapnext_step_256_w16.onnx.part* > models/tapnext_step_256_w16.onnx
(cd models && sha256sum -c tapnext_step_256_w16.onnx.sha256)
```

On Windows: `copy /b models\tapnext_step_256_w16.onnx.part00+models\tapnext_step_256_w16.onnx.part01+models\tapnext_step_256_w16.onnx.part02+models\tapnext_step_256_w16.onnx.part03+models\tapnext_step_256_w16.onnx.part04 models\tapnext_step_256_w16.onnx`

## The graph: one frame per call, recurrent state in/out

TAPNext is a recurrent model (RG-LRU + causal conv per token, plus ViT attention across tokens). The graph runs one frame and passes the state explicitly. `Q` is the number of point slots and is dynamic (1..4096).

| input | shape | meaning |
|---|---|---|
| `frame` | `[1,256,256,3]` f32 | RGB, HWC, values in **[-1, 1]** (`rgb/255*2-1`) |
| `query_points` | `[1,Q,3]` f32 | `(t, x, y)`. `t` is the absolute frame index where the point is defined. `x,y` are in 256-px coords. `t` > current step means the slot is not active yet. |
| `step` | `[1]` f32 | current frame index (0, 1, 2, …) |
| `video_reset` | `[1]` f32 | 1 on the first frame, else 0 |
| `point_reset` | `[1,Q]` f32 | 1 to wipe a slot's history when you recycle it, else 0 |
| `lru_state` | `[12,1024+Q,768]` f32 | zeros at start, then feed back `lru_state_out` |
| `conv_state` | `[12,1024+Q,3,768]` f32 | zeros at start, then feed back `conv_state_out` |

| output | shape | meaning |
|---|---|---|
| `tracks_xy` | `[1,Q,2]` | position `(x,y)` in 256-px coords |
| `visible_prob` | `[1,Q]` | sigmoid of the visibility logit |
| `certainty` | `[1,Q]` | probability mass within 8 px of the prediction |
| `lru_state_out`, `conv_state_out` | same as the inputs | next state |

Treat a point as reliable when `visible_prob * certainty > 0.5`. This is the official "use_certainty" rule.

To add a point at frame `t`, write `(t, x, y)` into a free slot. If that slot has been used before, also set `point_reset=1` for that one call. Ignore the outputs of inactive slots.

## Verification

- **Wrapper vs. original `TAPNext.forward`** (12 frames, points queried at frames 0 and 3, edge points): max difference 6e-5 px.
- **ONNX Runtime fp32 vs. original PyTorch:** max 1.4e-4 px. **w16 vs. original PyTorch:** max 0.016 px.
- **Synthetic object moving (14, 3) px/frame:** the C++ demo measures v = (14.0, 3.0).
- **`run_video.py` (ort-cpu) on a 1280×720 synthetic clip** with true motion (70, 8.44) px/frame and points added at frames 0, 4 and 6: measured velocities are 70.2 ± 0.3 and 8.5 ± 0.2.
- **TensorRT:** not tested here because this machine has no GPU. The C++ TensorRT backend compiles against the TensorRT 10.13 headers. `tools/build_trt_engine.py` checks the engine against ONNX Runtime after building.

## Quick start on a GPU machine

```bash
bash scripts/gpu_quickstart.sh path/to/gate_video.mp4 64
```

The script runs these steps in order:

1. Reassembles and checksums the ONNX.
2. Installs the Python dependencies.
3. Builds and verifies the TensorRT fp16 engine for Q=64.
4. Benchmarks ONNX Runtime CUDA against TensorRT.
5. Writes `<video>_tapnext_trt.mp4` and `.csv`.

The sections below explain each step.

## Windows (PowerShell, venv active)

Run everything from the repo root.

**1. Get the code.** Skip this if you already cloned the repo.

```powershell
git clone -b claude/magical-mccarthy-t34jfd https://github.com/haile-leul/tapnext_onnx
cd tapnext_onnx
```

**2. Install packages into the venv.** Use Python 3.10–3.12. Use the **same TensorRT version as your C++ project**, because an engine only loads in the TensorRT version that built it. The example below uses TensorRT 10.14.1.48 for CUDA 13; adjust the path to your install. `onnxruntime-gpu`, `torch` and TensorRT must all be CUDA 13 builds. Pip's default `torch` on Windows is CPU-only, so install it from the CUDA index.

```powershell
$env:TENSORRT_DIR = "D:\SVN\PTnD\extern\tensorrt_10.14.1.48_CUDA13"   # your C++ project's TensorRT
$env:TENSORRT_VERSION = "10.14.1.48"                                  # warn if Python loads another version

python -m pip install -U pip
pip uninstall -y onnxruntime torch tensorrt tensorrt-cu12 tensorrt-cu12-libs tensorrt-cu12-bindings tensorrt-cu13 tensorrt-cu13-libs tensorrt-cu13-bindings
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install onnxruntime-gpu onnx opencv-python numpy

# TensorRT Python bindings from the SAME zip (matching your Python, e.g. cp311):
pip install (Get-ChildItem "$env:TENSORRT_DIR\python\tensorrt-*-cp311-none-win_amd64.whl").FullName
#   (if that zip has no python\ folder: pip install tensorrt-cu13==10.14.1.48)

python tools\check_gpu_env.py   # must show tensorrt 10.14.1.48, "test session runs on: CUDAExecutionProvider", ALL GOOD
```

`TENSORRT_DIR` adds `<dir>\bin` and `<dir>\lib` to the DLL search path, so Python uses the same `nvinfer` DLLs as your C++ build. Set it in every new PowerShell window, or once with `setx TENSORRT_DIR ...` and `setx TENSORRT_VERSION ...`.

**3. Reassemble the ONNX and build the TensorRT engine.** The build takes a few minutes. Afterwards it prints how far the engine is from ONNX Runtime; expect well under 1 px.

```powershell
python tools\reassemble_model.py
python tools\build_trt_engine.py --onnx models\tapnext_step_256_w16.onnx --engine models\tapnext_step_256_fp16_q64.engine --min-q 64 --opt-q 64 --max-q 64
```

**4. Test speed and tracking quality.**

```powershell
# speed: ONNX Runtime CUDA vs TensorRT
python run_video.py --video C:\videos\gate.mp4 --backend ort-cuda --slots 64 --grid 8 --benchmark --max-frames 200
python run_video.py --video C:\videos\gate.mp4 --backend trt --model models\tapnext_step_256_fp16_q64.engine --slots 64 --grid 8 --benchmark --max-frames 200

# quality: writes C:\videos\gate_tapnext.mp4 (+ CSV)
python run_video.py --video C:\videos\gate.mp4 --backend trt --model models\tapnext_step_256_fp16_q64.engine --slots 64 --grid 8 --reseed-every 15 --reseed-grid 6 --drop-after 5 --csv C:\videos\gate_tracks.csv

# add points mid-video, or click them yourself on frames 0 and 30
python run_video.py --video C:\videos\gate.mp4 --backend trt --model models\tapnext_step_256_fp16_q64.engine --slots 64 --add "40:812,430;900,455" --add-grid "55:700,350,300,200:4"
python run_video.py --video C:\videos\gate.mp4 --backend trt --model models\tapnext_step_256_fp16_q64.engine --slots 64 --pick 0 --pick 30 --show
```

Steps 3–4 in one go: `.\scripts\gpu_quickstart.ps1 -Video C:\videos\gate.mp4 -Q 64 -TensorrtDir D:\SVN\PTnD\extern\tensorrt_10.14.1.48_CUDA13`.

**5. Check the engine with your C++ TensorRT install (no Python involved).** This loads the engine with the same DLLs your project uses and reports GPU latency:

```powershell
& "$env:TENSORRT_DIR\bin\trtexec.exe" --loadEngine=models\tapnext_step_256_fp16_q64.engine --shapes=query_points:1x64x3,point_reset:1x64,lru_state:12x1088x768,conv_state:12x1088x3x768 --useCudaGraph --noDataTransfers
```

(1088 = 1024 + Q.) If it loads here, it loads in your C++ project through `cpp/tapnext_trt_tracker.hpp`. In CMake, set `-DTENSORRT_ROOT=$env:TENSORRT_DIR`.

**Notes:**
- `--slots` must match the Q the engine was built for. To support a range, build with `--min-q 1 --opt-q 64 --max-q 256`.
- **If `ort-cuda` reports "GPU provider not available":** run `tools\check_gpu_env.py`. Usually `onnxruntime` and `onnxruntime-gpu` are both installed (uninstall both, then reinstall `onnxruntime-gpu`), or the CUDA/cuDNN DLLs can't be found. With the cu124 torch installed, `onnxruntime-gpu` ≥ 1.21 loads them automatically.
- **If `.ps1` scripts are blocked:** run `Set-ExecutionPolicy -Scope Process Bypass` first.

## TensorRT fp16 engine

A TensorRT engine only runs on the GPU architecture and TensorRT version it was built with, so build it on your machine:

```bash
pip install tensorrt onnx onnxruntime-gpu torch   # torch is only used for GPU buffers in Python
python tools/build_trt_engine.py --onnx models/tapnext_step_256_w16.onnx \
    --engine models/tapnext_step_256_fp16.engine --min-q 1 --opt-q 64 --max-q 256
```

The default `--precision mixed` runs most of the network in fp16. It pins these numerically sensitive parts to fp32:

- RMSNorm square → mean → rsqrt (x² can overflow fp16)
- LayerNorms
- RG-LRU decay and state update (a ≈ 1 − 1e-4 rounds to 1 in fp16)
- output heads, soft-argmax and certainty (coordinates need sub-pixel precision)

All of these are cheap elementwise or per-point ops.

After building, the script runs the engine and ONNX Runtime side by side on a synthetic clip and prints the largest position and visibility differences. Expect well under 1 px in 256-px coordinates. If it is much larger, compare against `--precision fp32`. `--precision fp16` lets TensorRT choose freely (fastest, but may drift).

Engine inputs and outputs have the same names, shapes and fp32 types as the ONNX graph. `Q` must lie within the `--min-q`..`--max-q` range. For the best speed with a fixed slot count, set all three to that value.

`trtexec --onnx=models/tapnext_step_256_w16.onnx --fp16 --saveEngine=...` with matching `--minShapes/--optShapes/--maxShapes` also works, but trtexec does not pin the fp32 layers. Check an engine built that way with `python tools/build_trt_engine.py --verify-only --engine <file>`.

## Testing on a video (Python, GPU)

```bash
# onnxruntime-gpu (CUDA EP); the state stays in GPU memory
python run_video.py --video gate.mp4 --backend ort-cuda --grid 8

# native TensorRT engine, ROI on the lanes, fresh 6x6 grid every 15 frames, drop points that stay lost
python run_video.py --video gate.mp4 --backend trt --model models/tapnext_step_256_fp16.engine \
    --roi 400,200,1100,800 --reseed-every 15 --reseed-grid 6 --drop-after 5 --csv tracks.csv

# add points mid-video: 2 points at frame 40, a 4x4 grid in a box at frame 55
python run_video.py --video gate.mp4 --backend ort-cuda \
    --add "40:812,430;900,455" --add-grid "55:700,350,300,200:4"

# click points yourself on frames 0 and 30 (needs a display)
python run_video.py --video gate.mp4 --backend ort-cuda --pick 0 --pick 30

# timing only
python run_video.py --video gate.mp4 --backend trt --model models/tapnext_step_256_fp16.engine --benchmark
```

- **Backends:**
  - `ort-cpu`
  - `ort-cuda` (needs `onnxruntime-gpu`)
  - `ort-trt` (ONNX Runtime's TensorRT EP, fp16; the engine is cached next to the model)
  - `trt` (`.engine` file; needs `tensorrt` + `torch`)
- **Coordinates:** all coordinates are in original video pixels.
- **Point modes:** `--grid`, `--add`, `--add-grid`, `--points-json`, `--pick`, `--reseed-every`.
- **Slots:** `--slots` is the fixed slot count `Q` (default 64). When all slots are busy, the oldest point is evicted. `--drop-after N` frees a slot after N unreliable frames in a row.
- **Output video:**
  - one colour per point with a trail
  - a white arrow for velocity (`--vel-scale`)
  - unreliable points are hidden unless you pass `--show-unreliable`
  - the ROI is drawn as a box
  - a header line shows model latency and FPS
- **Timing:** reported after a warm-up as model mean/p50/p95 ms and FPS, plus preprocessing and end-to-end time.
- **CSV:** `--csv` writes `frame, point_id, slot, x, y, vx, vy, visible, certainty, reliable` for every active point.

## C++

```bash
cmake -S cpp -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0 \
      -DWITH_TENSORRT=ON -DTENSORRT_ROOT=/path/to/TensorRT-10.x
cmake --build build
./build/tapnext_demo models/tapnext_step_256_fp16.engine video.mp4      # TensorRT
./build/tapnext_demo models/tapnext_step_256_w16.onnx video.mp4 [cuda]  # ONNX Runtime
```

`TapNextTrtTracker` (`cpp/tapnext_trt_tracker.hpp`) and `TapNextTracker` (`cpp/tapnext_tracker.hpp`) share the API in `cpp/tapnext_points.hpp`:

- `addPoint`, `removePoint`
- `step`, which returns `x, y, vx, vy, visible, certainty` in original image pixels
- `medianVelocity(box)`

The TensorRT version keeps the recurrent state (about 150 MB at Q=64) in two GPU buffers that swap each frame. Only the 256×256 frame, the queries and the Q outputs cross PCIe. The ONNX Runtime C++ version keeps the state in host memory, which is fine for CPU but slow on GPU.

Feed a fixed ROI around the gate lanes instead of the whole frame. The ROI is resized to 256×256, so a tighter ROI gives more effective resolution.

## Rebuilding

```bash
pip install torch torchvision einops onnx onnxscript onnxruntime
git clone https://github.com/google-deepmind/tapnet
wget https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
python export_tapnext_onnx.py --tapnet_root tapnet --ckpt bootstapnext_ckpt.npz --out tapnext_step_256.onnx
python tools/convert_fp16.py tapnext_step_256.onnx tapnext_step_256_w16.onnx
python tools/check_onnx.py tapnet bootstapnext_ckpt.npz tapnext_step_256.onnx
```

The model and weights are Apache 2.0, from google-deepmind/tapnet.
