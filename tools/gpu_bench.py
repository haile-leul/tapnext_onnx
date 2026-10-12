"""Where does the time go? GPU raw speed vs TAPNext (official forward and the export wrapper).

  python tools/gpu_bench.py                                   # GPU matmul speed + memory only
  python tools/gpu_bench.py --tapnet-root ..\\tapnet --ckpt ..\\bootstapnext_ckpt.npz

Prints:
  1. free/total VRAM and raw matmul TFLOPS (fp32 / fp16). A GTX 1650 should
     reach ~2.5-3 TFLOPS fp32 and ~5 fp16. Far less => the GPU is throttled,
     shared, or spilling into system RAM (Windows "CUDA sysmem fallback").
  2. TAPNext per-frame latency (357 GFLOP/frame at Q=64), timed with
     torch.cuda.synchronize, for the OFFICIAL TAPNext.forward and for the
     wrapper that was exported to ONNX, fp32 and fp16 autocast, + peak VRAM.
  3. the top CUDA kernels of one wrapper step (torch profiler).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GFLOP_PER_FRAME = 357.2  # measured with torch FlopCounterMode at Q=64


def timed(fn, iters, warmup=3):
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  ts = []
  for _ in range(iters):
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1000)
  return np.array(ts)


def report(name, ts, gflop=GFLOP_PER_FRAME):
  m = float(np.median(ts))
  print(f'  {name:42s} median {m:8.1f} ms  ({1000 / m:5.1f} fps, {gflop / m:5.2f} TFLOPS effective)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--tapnet-root')
  ap.add_argument('--ckpt')
  ap.add_argument('--q', type=int, default=64)
  ap.add_argument('--iters', type=int, default=10)
  args = ap.parse_args()

  if not torch.cuda.is_available():
    sys.exit('CUDA not available in torch')
  dev = torch.device('cuda')
  p = torch.cuda.get_device_properties(0)
  free, total = torch.cuda.mem_get_info()
  print(f'GPU: {p.name}, {p.multi_processor_count} SMs, CC {p.major}.{p.minor}, '
        f'VRAM free {free / 2**30:.2f} / {total / 2**30:.2f} GB')
  if free < 1.5 * 2**30:
    print('  WARNING: little free VRAM - other processes are using the GPU '
          '(check nvidia-smi); on Windows the driver then silently spills to system RAM')

  print('\n1) raw matmul speed (4096x4096)')
  for dt in (torch.float32, torch.float16):
    a = torch.randn(4096, 4096, device=dev, dtype=dt)
    b = torch.randn(4096, 4096, device=dev, dtype=dt)
    report(f'matmul {str(dt).split(".")[1]}', timed(lambda: a @ b, args.iters), 2 * 4096**3 / 1e9)
    del a, b
  torch.cuda.empty_cache()

  if not (args.tapnet_root and args.ckpt):
    print('\n(pass --tapnet-root and --ckpt to also time TAPNext)')
    return

  from export_tapnext_onnx import TAPNextStep, load_model, NUM_LAYERS, NUM_PATCHES, WIDTH  # pylint: disable=g-import-not-at-top
  model = load_model(args.tapnet_root, args.ckpt).to(dev)
  q = args.q
  rng = np.random.default_rng(0)
  frame = torch.rand(1, 1, 256, 256, 3, device=dev) * 2 - 1
  qyx = torch.from_numpy(np.concatenate([np.zeros((1, q, 1)), rng.uniform(5, 250, (1, q, 2))], -1)
                         .astype(np.float32)).to(dev)

  print(f'\n2) TAPNext one online step, Q={q} ({GFLOP_PER_FRAME:.0f} GFLOP/frame)')
  for fp16 in (False, True):
    tag = 'fp16 autocast' if fp16 else 'fp32'
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
      _, _, _, state = model(video=frame, query_points=qyx)
      torch.cuda.reset_peak_memory_stats()
      ts = timed(lambda: model(video=frame, state=state), args.iters)
      report(f'official TAPNext.forward ({tag})', ts)
    print(f'    peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB')

  step = TAPNextStep(model).eval()
  n = NUM_PATCHES + q
  ins = (frame[0], torch.cat([qyx[..., :1], qyx[..., 2:3], qyx[..., 1:2]], -1),
         torch.zeros(1, device=dev), torch.zeros(1, device=dev), torch.zeros(1, q, device=dev),
         torch.zeros(NUM_LAYERS, n, WIDTH, device=dev), torch.zeros(NUM_LAYERS, n, 3, WIDTH, device=dev))
  for fp16 in (False, True):
    tag = 'fp16 autocast' if fp16 else 'fp32'
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
      torch.cuda.reset_peak_memory_stats()
      report(f'export wrapper TAPNextStep ({tag})', timed(lambda: step(*ins), args.iters))
    print(f'    peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB')

  print('\n3) top CUDA kernels, one wrapper step (fp16 autocast)')
  from torch.profiler import ProfilerActivity, profile  # pylint: disable=g-import-not-at-top
  with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      step(*ins)
      torch.cuda.synchronize()
  print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=12))


if __name__ == '__main__':
  main()
