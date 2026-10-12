# TAPNext (BootsTAPNext, 256×256) → ONNX

This is a streaming ONNX export of [BootsTAPNext](https://github.com/google-deepmind/tapnet), the online point tracker from google-deepmind/tapnet. It uses the official `bootstapnext_ckpt.npz` checkpoint with the TRecViT-B/8 backbone at 256×256. It also includes a C++ (ONNX Runtime + OpenCV) wrapper that returns per-point positions and velocities.

## Files

| file | what |
|---|---|
| `models/tapnext_step_256_w16.onnx.part00..04` (390 MB total, split for GitHub's 100 MB limit) | Weights are stored as fp16 and cast to fp32 inside the graph. All compute and I/O run in fp32. Max deviation from fp32 is 0.016 px. |
| `export_tapnext_onnx.py` | Rebuilds the full fp32 model (`tapnext_step_256.onnx`, 779 MB) from the checkpoint. |
| `tools/convert_fp16.py` | Converts the fp32 model into the `w16` file. |
| `tools/check_parity_torch.py`, `tools/check_onnx.py`, `tools/check_midstream.py` | Checks against the original PyTorch TAPNext. |
| `cpp/tapnext_tracker.hpp`, `cpp/demo.cpp`, `cpp/CMakeLists.txt` | C++ wrapper and demo. |

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

## C++

```bash
cmake -S cpp -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0
cmake --build build
./build/tapnext_demo models/tapnext_step_256_w16.onnx video.mp4 [cuda]
```

See `cpp/tapnext_tracker.hpp` for `addPoint`, `removePoint`, `step` (which returns `x, y, vx, vy, visible, certainty` in original image pixels) and `medianVelocity(box)`.

Performance notes:
- Each frame moves about 150 MB of state (Q≈64). On GPU, bind the state tensors to device memory (IoBinding with CUDA memory, ping-ponged) so the state never round-trips to the host.
- For fp16 compute, give the fp32 or w16 model to TensorRT (`trtexec --fp16`, or the ORT TensorRT EP with `trt_fp16_enable`).
- Feed a fixed ROI around the gate lanes instead of the whole frame. The ROI is resized to 256×256, so a tighter ROI gives more effective resolution.

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
