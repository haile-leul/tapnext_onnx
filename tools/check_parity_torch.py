"""Compare the export wrapper (TAPNextStep) with the original TAPNext torch model."""
import sys
import numpy as np
import torch

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export_tapnext_onnx import TAPNextStep, load_model, NUM_LAYERS, NUM_PATCHES, WIDTH  # noqa
from tools.make_test_video import make_video  # noqa

tapnet_root, ckpt = sys.argv[1], sys.argv[2]
model = load_model(tapnet_root, ckpt)
step = TAPNextStep(model).eval()
video, tl = make_video()
# queries (t, x, y): points on the object at frame 0, plus 2 points queried at frame 3
pts = [(0, 40, 110), (0, 70, 130), (0, 100, 150), (0, 120, 100), (0, 200, 30), (3, tl[3, 0] + 30, tl[3, 1] + 20), (3, tl[3, 0] + 90, tl[3, 1] + 70), (0, 0.2, 255.7), (0, 255.9, 0.3), (0, 33.37, 177.81)]
qxy = torch.tensor(pts, dtype=torch.float32)[None]
qyx = qxy[..., [0, 2, 1]]
q = qxy.shape[1]
vt = torch.from_numpy(video)[None]

lru = torch.zeros(NUM_LAYERS, NUM_PATCHES + q, WIDTH)
conv = torch.zeros(NUM_LAYERS, NUM_PATCHES + q, 3, WIDTH)
max_err = 0
with torch.no_grad():
  state = None
  for t in range(video.shape[0]):
    if state is None:
      tr, _, vl, state = model(video=vt[:, :1], query_points=qyx)
    else:
      tr, _, vl, state = model(video=vt[:, t:t + 1], state=state)
    o = step(vt[0, t:t + 1], qxy, torch.tensor([float(t)]), torch.tensor([1.0 if t == 0 else 0.0]), torch.zeros(1, q), lru, conv)
    lru, conv = o[3], o[4]
    ref_xy = tr[0, 0][:, [1, 0]]
    err = (o[0][0] - ref_xy).abs().max().item()
    verr = (o[1][0] - torch.sigmoid(vl[0, 0, :, 0])).abs().max().item()
    max_err = max(max_err, err)
    print(f't={t} max|dxy|={err:.2e} max|dvis|={verr:.2e}  pred={o[0][0][:3].numpy().round(1).tolist()} obj_tl={tl[t].tolist()}')
print('MAX', max_err)
