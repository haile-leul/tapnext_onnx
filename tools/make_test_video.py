"""Synthetic test clip: a textured 'vehicle' patch moving fast over a textured background."""
import numpy as np
import torch
import torch.nn.functional as F


def _texture(h, w, seed):
  g = torch.Generator().manual_seed(seed)
  out = torch.zeros(3, h, w)
  for s in (4, 8, 16, 32):
    n = torch.rand(1, 3, h // s + 1, w // s + 1, generator=g)
    out += F.interpolate(n, size=(h, w), mode='bicubic', align_corners=False)[0]
  out = (out - out.min()) / (out.max() - out.min())
  return out.permute(1, 2, 0).numpy()  # [h, w, 3] in [0, 1]


def make_video(num_frames=12, velocity=(14.0, 3.0), size=256):
  """Returns video [T, 256, 256, 3] in [-1, 1] and the object's top-left per frame (x, y)."""
  bg = _texture(size, size, 0) * 0.6
  obj = _texture(96, 128, 1)
  frames, tl = [], []
  for t in range(num_frames):
    x0 = int(round(10 + velocity[0] * t))
    y0 = int(round(80 + velocity[1] * t))
    f = bg.copy()
    xa, xb = max(x0, 0), min(x0 + 128, size)
    ya, yb = max(y0, 0), min(y0 + 96, size)
    if xa < xb and ya < yb:
      f[ya:yb, xa:xb] = obj[ya - y0:yb - y0, xa - x0:xb - x0]
    frames.append(f)
    tl.append((x0, y0))
  video = np.stack(frames).astype(np.float32) * 2 - 1
  return video, np.array(tl, np.float32)
