"""Backends for the TAPNext step model + a point-slot manager.

Backends (all expose .step(frame, query_points, step, video_reset, point_reset)
-> (tracks_xy [Q,2], visible_prob [Q], certainty [Q]) in 256-px coords, and
keep the recurrent state internally, on the GPU for GPU backends):

  ort-cpu   onnxruntime, CPU
  ort-cuda  onnxruntime-gpu, CUDA EP, state ping-ponged in GPU memory (IOBinding)
  ort-trt   onnxruntime-gpu, TensorRT EP (fp16, engine cached in --trt-cache)
  trt       native TensorRT engine (.engine from tools/build_trt_engine.py);
            needs `tensorrt` and `torch` (CUDA) for device buffers
"""

import os
import numpy as np

NUM_LAYERS, NUM_PATCHES, WIDTH, IMG = 12, 1024, 768, 256
INACTIVE_T = 1e9


def _state_shapes(q):
  n = NUM_PATCHES + q
  return (NUM_LAYERS, n, WIDTH), (NUM_LAYERS, n, 3, WIDTH)


class OrtBackend:
  """onnxruntime backend; the recurrent state never leaves the device."""

  def __init__(self, model_path, num_slots, provider='cpu', trt_cache=None):
    import onnxruntime as ort  # pylint: disable=g-import-not-at-top
    self.q = num_slots
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == 'cpu':
      providers, self.device = ['CPUExecutionProvider'], 'cpu'
    elif provider == 'cuda':
      providers, self.device = ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'cuda'
    elif provider == 'trt':
      cache = trt_cache or os.path.join(os.path.dirname(os.path.abspath(model_path)), 'ort_trt_cache')
      os.makedirs(cache, exist_ok=True)
      lru_s, conv_s = _state_shapes(num_slots)
      shapes = (f'query_points:1x{num_slots}x3,point_reset:1x{num_slots},'
                f'lru_state:{"x".join(map(str, lru_s))},conv_state:{"x".join(map(str, conv_s))}')
      providers = [('TensorrtExecutionProvider', {
          'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
          'trt_engine_cache_path': cache, 'trt_max_workspace_size': 4 << 30,
          'trt_profile_min_shapes': shapes, 'trt_profile_max_shapes': shapes,
          'trt_profile_opt_shapes': shapes}),
                   'CUDAExecutionProvider', 'CPUExecutionProvider']
      self.device = 'cuda'
    else:
      raise ValueError(provider)
    self.sess = ort.InferenceSession(model_path, so, providers=providers)
    self.name = f'ort-{provider} ({self.sess.get_providers()[0]})'
    if provider != 'cpu' and self.sess.get_providers()[0] == 'CPUExecutionProvider':
      raise RuntimeError('GPU provider not available; install onnxruntime-gpu matching your CUDA')
    self._ort = ort
    self.reset()

  def reset(self):
    lru_s, conv_s = _state_shapes(self.q)
    mk = lambda s: self._ort.OrtValue.ortvalue_from_numpy(np.zeros(s, np.float32), self.device, 0)
    self.lru = [mk(lru_s), mk(lru_s)]
    self.conv = [mk(conv_s), mk(conv_s)]
    self.ping = 0

  def step(self, frame, query_points, step, video_reset, point_reset):
    cur, nxt = self.ping, 1 - self.ping
    io = self.sess.io_binding()
    io.bind_cpu_input('frame', np.ascontiguousarray(frame, np.float32))
    io.bind_cpu_input('query_points', np.ascontiguousarray(query_points, np.float32))
    io.bind_cpu_input('step', np.array([step], np.float32))
    io.bind_cpu_input('video_reset', np.array([video_reset], np.float32))
    io.bind_cpu_input('point_reset', np.ascontiguousarray(point_reset, np.float32))
    io.bind_ortvalue_input('lru_state', self.lru[cur])
    io.bind_ortvalue_input('conv_state', self.conv[cur])
    for name in ('tracks_xy', 'visible_prob', 'certainty'):
      io.bind_output(name, 'cpu')
    io.bind_ortvalue_output('lru_state_out', self.lru[nxt])
    io.bind_ortvalue_output('conv_state_out', self.conv[nxt])
    self.sess.run_with_iobinding(io)
    outs = io.get_outputs()
    self.ping = nxt
    return outs[0].numpy()[0], outs[1].numpy()[0], outs[2].numpy()[0]


class TrtBackend:
  """Native TensorRT engine; device buffers via torch."""

  def __init__(self, engine_path, num_slots):
    import tensorrt as trt  # pylint: disable=g-import-not-at-top
    import torch  # pylint: disable=g-import-not-at-top
    self.torch, self.q = torch, num_slots
    self.name = f'trt ({trt.__version__})'
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
      self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if self.engine is None:
      raise RuntimeError('could not load engine (built for another GPU / TensorRT version?)')
    self.ctx = self.engine.create_execution_context()
    self.stream = torch.cuda.Stream()
    lru_s, conv_s = _state_shapes(num_slots)
    q = num_slots
    self.in_shapes = {
        'frame': (1, IMG, IMG, 3), 'query_points': (1, q, 3), 'step': (1,),
        'video_reset': (1,), 'point_reset': (1, q), 'lru_state': lru_s,
        'conv_state': conv_s}
    for name, shape in self.in_shapes.items():
      if not self.ctx.set_input_shape(name, shape):
        raise RuntimeError(f'engine profile does not allow {name}{shape}: rebuild with a '
                           f'--min-q/--max-q range that includes {q}')
    dev = 'cuda'
    z = lambda s: torch.zeros(s, dtype=torch.float32, device=dev)
    self.buf = {n: z(s) for n, s in self.in_shapes.items() if 'state' not in n}
    self.buf['tracks_xy'] = z((1, q, 2))
    self.buf['visible_prob'] = z((1, q))
    self.buf['certainty'] = z((1, q))
    self.lru = [z(lru_s), z(lru_s)]
    self.conv = [z(conv_s), z(conv_s)]
    self.ping = 0
    self.host = {n: torch.zeros(self.buf[n].shape, dtype=torch.float32).pin_memory()
                 for n in ('frame', 'query_points', 'step', 'video_reset', 'point_reset',
                           'tracks_xy', 'visible_prob', 'certainty')}

  def reset(self):
    for t in self.lru + self.conv:
      t.zero_()
    self.ping = 0

  def step(self, frame, query_points, step, video_reset, point_reset):
    torch = self.torch
    h = self.host
    h['frame'].numpy()[...] = frame
    h['query_points'].numpy()[...] = query_points
    h['step'].numpy()[0] = step
    h['video_reset'].numpy()[0] = video_reset
    h['point_reset'].numpy()[...] = point_reset
    cur, nxt = self.ping, 1 - self.ping
    with torch.cuda.stream(self.stream):
      for n in ('frame', 'query_points', 'step', 'video_reset', 'point_reset'):
        self.buf[n].copy_(h[n], non_blocking=True)
      addr = {n: t.data_ptr() for n, t in self.buf.items()}
      addr.update(lru_state=self.lru[cur].data_ptr(), conv_state=self.conv[cur].data_ptr(),
                  lru_state_out=self.lru[nxt].data_ptr(),
                  conv_state_out=self.conv[nxt].data_ptr())
      for n, a in addr.items():
        self.ctx.set_tensor_address(n, a)
      self.ctx.execute_async_v3(self.stream.cuda_stream)
      for n in ('tracks_xy', 'visible_prob', 'certainty'):
        h[n].copy_(self.buf[n], non_blocking=True)
    self.stream.synchronize()
    self.ping = nxt
    return h['tracks_xy'].numpy()[0].copy(), h['visible_prob'].numpy()[0].copy(), \
        h['certainty'].numpy()[0].copy()


def make_backend(kind, model_path, num_slots, trt_cache=None):
  if kind == 'trt':
    return TrtBackend(model_path, num_slots)
  return OrtBackend(model_path, num_slots, provider=kind.split('-', 1)[1], trt_cache=trt_cache)


class PointTracker:
  """Slot pool + ROI mapping + velocities, in original image pixels.

  Mirrors cpp/tapnext_tracker.hpp.
  """

  def __init__(self, backend, num_slots, roi):
    self.b, self.q = backend, num_slots
    self.roi = roi  # (x, y, w, h) in original pixels
    self.queries = np.zeros((1, num_slots, 3), np.float32)
    self.queries[0, :, 0] = INACTIVE_T
    self.reset_mask = np.zeros((1, num_slots), np.float32)
    self.used = np.zeros(num_slots, bool)
    self.active = np.zeros(num_slots, bool)
    self.fresh = np.zeros(num_slots, bool)
    self.born = np.zeros(num_slots, np.int64)   # frame the point was added
    self.uid = np.full(num_slots, -1, np.int64)  # unique point id (for colours / CSV)
    self.next_uid = 0
    self.xy = np.zeros((num_slots, 2), np.float32)
    self.v = np.zeros((num_slots, 2), np.float32)
    self.vis = np.zeros(num_slots, np.float32)
    self.cert = np.zeros(num_slots, np.float32)
    self.bad_streak = np.zeros(num_slots, np.int64)
    self.t = 0

  def to_model(self, x, y):
    rx, ry, rw, rh = self.roi
    return (x - rx) * IMG / rw, (y - ry) * IMG / rh

  def add(self, x, y, evict_oldest=True):
    """Adds a point (original px) starting at the next step(). Returns slot or -1."""
    free = np.flatnonzero(~self.active)
    if free.size:
      # prefer never-used slots (their history is the model's native 'unknown' tokens)
      never = free[~self.used[free]]
      s = int(never[0] if never.size else free[0])
    elif evict_oldest:
      s = int(np.argmin(np.where(self.active, self.born, np.iinfo(np.int64).max)))
    else:
      return -1
    mx, my = self.to_model(x, y)
    self.queries[0, s] = (self.t, mx, my)
    self.reset_mask[0, s] = 1.0 if self.used[s] else 0.0
    self.used[s] = self.active[s] = self.fresh[s] = True
    self.born[s] = self.t
    self.uid[s] = self.next_uid
    self.next_uid += 1
    self.xy[s] = (x, y)
    self.v[s] = 0
    self.bad_streak[s] = 0
    return s

  def remove(self, s):
    self.active[s] = False
    self.queries[0, s, 0] = INACTIVE_T

  def step(self, frame_rgb_m11):
    """frame_rgb_m11: [256,256,3] float32 RGB in [-1, 1] (already ROI-cropped)."""
    xy, vis, cert = self.b.step(frame_rgb_m11[None], self.queries, float(self.t),
                                float(self.t == 0), self.reset_mask)
    self.reset_mask[:] = 0
    rx, ry, rw, rh = self.roi
    new = np.stack([rx + xy[:, 0] * rw / IMG, ry + xy[:, 1] * rh / IMG], -1)
    a = self.active
    self.v[a & ~self.fresh] = (new - self.xy)[a & ~self.fresh]
    self.fresh[:] = False
    self.xy[a] = new[a]
    self.vis[a], self.cert[a] = vis[a], cert[a]
    reliable = self.vis * self.cert > 0.5
    self.bad_streak = np.where(a & ~reliable, self.bad_streak + 1, 0)
    self.t += 1
    return reliable
