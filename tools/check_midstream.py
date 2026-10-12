"""Check points added mid-stream: into a never-used slot, and into a recycled slot (point_reset=1)."""
import os
import sys
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export_tapnext_onnx import NUM_LAYERS, NUM_PATCHES, WIDTH  # noqa
from tools.make_test_video import make_video  # noqa

sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
video, tl = make_video()
q = 3
qp = np.array([[[0, tl[0, 0] + 20, tl[0, 1] + 20],      # slot 0: object point from frame 0
                [1e9, 0, 0],                          # slot 1: unused until frame 5
                [0, 230, 30]]], np.float32)           # slot 2: background point
lru = np.zeros((NUM_LAYERS, NUM_PATCHES + q, WIDTH), np.float32)
conv = np.zeros((NUM_LAYERS, NUM_PATCHES + q, 3, WIDTH), np.float32)
for t in range(video.shape[0]):
  reset = np.zeros((1, q), np.float32)
  if t == 5:
    qp[0, 1] = [5, tl[5, 0] + 60, tl[5, 1] + 50]       # first use of slot 1
    qp[0, 2] = [5, tl[5, 0] + 100, tl[5, 1] + 30]      # recycle slot 2 onto the object
    reset[0, 2] = 1
  xy, vis, cert, lru, conv = sess.run(None, {
      'frame': video[t:t + 1], 'query_points': qp, 'step': np.array([t], np.float32),
      'video_reset': np.array([float(t == 0)], np.float32), 'point_reset': reset,
      'lru_state': lru, 'conv_state': conv})
  off = xy[0] - tl[t]  # position relative to the object's top-left: constant if tracked
  print(t, np.round(off, 1).tolist(), np.round(vis[0] * cert[0], 2).tolist())
