"""Run the exported ONNX step model in onnxruntime and compare with the original torch TAPNext."""
import os
import sys
import time
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export_tapnext_onnx import load_model, NUM_LAYERS, NUM_PATCHES, WIDTH  # noqa
from tools.make_test_video import make_video  # noqa

tapnet_root, ckpt, onnx_path = sys.argv[1:4]
model = load_model(tapnet_root, ckpt)
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
video, tl = make_video()
pts = [(0, 40, 110), (0, 70, 130), (0, 100, 150), (0, 120, 100), (0, 200, 30),
       (3, tl[3, 0] + 30, tl[3, 1] + 20), (3, tl[3, 0] + 90, tl[3, 1] + 70),
       (0, 0.2, 255.7), (0, 33.37, 177.81), (100, 5, 5)]
qxy = np.array(pts, np.float32)[None]
q = qxy.shape[1]
lru = np.zeros((NUM_LAYERS, NUM_PATCHES + q, WIDTH), np.float32)
conv = np.zeros((NUM_LAYERS, NUM_PATCHES + q, 3, WIDTH), np.float32)
vt = torch.from_numpy(video)[None]
worst = 0
with torch.no_grad():
  state = None
  for t in range(video.shape[0]):
    qyx = torch.from_numpy(qxy[..., [0, 2, 1]])
    if state is None:
      tr, tlog, vl, state = model(video=vt[:, :1], query_points=qyx)
    else:
      tr, tlog, vl, state = model(video=vt[:, t:t + 1], state=state)
    t0 = time.time()
    xy, vis, cert, lru, conv = sess.run(None, {
        'frame': video[t:t + 1], 'query_points': qxy,
        'step': np.array([t], np.float32),
        'video_reset': np.array([1.0 if t == 0 else 0.0], np.float32),
        'point_reset': np.zeros((1, q), np.float32),
        'lru_state': lru, 'conv_state': conv})
    dt = time.time() - t0
    ref = tr[0, 0][:, [1, 0]].numpy()
    e = np.abs(xy[0] - ref)[:-1].max()  # last point is not queried yet (t=100)
    ev = np.abs(vis[0] - torch.sigmoid(vl[0, 0, :, 0]).numpy()).max()
    worst = max(worst, e)
    print(f't={t} {dt*1000:.0f} ms  max|dxy|={e:.2e} max|dvis|={ev:.2e} '
          f'p0={xy[0,0].round(1)} vis0={vis[0,0]:.2f} cert0={cert[0,0]:.2f} obj_tl={tl[t]}')
print('WORST', worst)
