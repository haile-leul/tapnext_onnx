"""Is the GPU delivering? Raw matmul TFLOPS vs Track-On2 encoder / tracker time (PyTorch).

  python tools/gpu_bench.py
  python tools/gpu_bench.py --trackon-root ../track_on --ckpt ../trackon2_dinov3_checkpoint.pt

Track-On2 costs ~140 GFLOP per frame for the encoder plus ~30 GFLOP + 0.44 GFLOP
per point for the tracker (measured with torch FlopCounterMode). A GTX 1650
should reach ~2.5-3 TFLOPS fp32 / ~5 fp16 on the matmul test; much less means
the GPU is throttled, shared, or spilling VRAM into system RAM (Windows
"CUDA - Sysmem Fallback Policy").
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def timed(fn, iters=10, warmup=3):
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  ts = []
  for _ in range(iters):
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1000)
  return float(np.median(ts))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--trackon-root')
  ap.add_argument('--ckpt')
  ap.add_argument('--points', type=int, nargs='+', default=[16, 64, 256])
  args = ap.parse_args()
  if not torch.cuda.is_available():
    sys.exit('CUDA not available in torch')
  p = torch.cuda.get_device_properties(0)
  free, total = torch.cuda.mem_get_info()
  print(f'GPU {p.name}, {p.multi_processor_count} SMs, VRAM free {free / 2**30:.2f}/{total / 2**30:.2f} GB')
  for dt in (torch.float32, torch.float16):
    a = torch.randn(4096, 4096, device='cuda', dtype=dt)
    ms = timed(lambda: a @ a)
    print(f'  matmul {str(dt)[6:]:8s} {ms:7.1f} ms  -> {2 * 4096**3 / ms / 1e9:.2f} TFLOPS')
    del a
  torch.cuda.empty_cache()
  if not args.trackon_root:
    return
  from trackon_export.loading import build_model  # pylint: disable=g-import-not-at-top
  m, _ = build_model(args.trackon_root, args.ckpt)
  m.cuda()
  frame = torch.rand(1, 3, 384, 512, device='cuda') * 255
  for fp16 in (False, True):
    tag = 'fp16 autocast' if fp16 else 'fp32'
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
      ms = timed(lambda: m.extract_frame_features(frame))
      print(f'  encoder ({tag}) {ms:7.1f} ms  ({140 / ms:.2f} TFLOPS effective)')
      feats = m.extract_frame_features(frame)
      for n in args.points:
        q = torch.randn(n, m.D, device='cuda')
        mask = torch.zeros(n, m.M, dtype=torch.bool, device='cuda')
        mem = torch.randn(n, m.M, m.D, device='cuda')
        ms = timed(lambda: m.track_frame(q, mask, mem, feats, 384, 512))
        print(f'  tracker ({tag}) N={n:4d} {ms:7.1f} ms')


if __name__ == '__main__':
  main()
