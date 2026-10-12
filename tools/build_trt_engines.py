"""Build the TensorRT engines (encoder + tracker) from the ONNX models and verify them.

  python tools/build_trt_engines.py                      # fp16 "mixed", points 1..1024
  python tools/build_trt_engines.py --max-points 4096 --opt-points 128
  python tools/build_trt_engines.py --verify-only        # re-check existing engines

Writes models/trackon_encoder.engine and models/trackon_tracker.engine. Engines
only load in the TensorRT version (and GPU architecture) that built them: run
this with the TensorRT your C++ project uses (set TENSORRT_DIR, see README).

Precision (--precision, per engine with --encoder-precision/--tracker-precision):
  mixed  fp16, but numerically sensitive layers pinned to fp32: all LayerNorm /
         InstanceNorm (GroupNorm), and in the tracker the coordinate maths
         (argmax/top-k -> pixel coordinates -> sampling grids -> output points),
         which needs better than fp16's 0.25 px resolution at x ~ 512.
  fp16   TensorRT decides everything (fastest, may drift).
  fp32   reference.

After building, the engines run next to onnxruntime (fp32) on a short clip
(points added mid-way, some removed) and the script prints the max position /
visibility difference, NaN counts and timings. It exits with an error on NaN.
"""

import argparse
import collections
import os
import sys

import numpy as np
import onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from trackon_runtime import import_tensorrt, make_backend  # pylint: disable=g-import-not-at-top

TAG = 'fp32__'
ELEMENTWISE = {'Add', 'Sub', 'Mul', 'Div', 'Floor', 'Clip', 'Cast', 'Concat', 'Reshape', 'Expand',
               'Unsqueeze', 'Squeeze', 'Gather', 'Slice', 'Transpose', 'Where', 'Neg', 'Split'}


def tag_fp32_nodes(model, coordinates):
  """Prefix the names of ONNX nodes that must stay fp32 with TAG. Returns count."""
  g = model.graph
  cons = collections.defaultdict(list)
  prod = {}
  for n in g.node:
    for i in n.input:
      cons[i].append(n)
    for o in n.output:
      prod[o] = n
  tagged = set()
  for n in g.node:
    if n.op_type in ('LayerNormalization', 'InstanceNormalization', 'GroupNormalization',
                     'Sin', 'Cos'):
      tagged.add(id(n))

  def forward(starts):
    stack = list(starts)
    while stack:
      t = stack.pop()
      for c in cons[t]:
        if c.op_type in ELEMENTWISE and id(c) not in tagged:
          tagged.add(id(c))
          stack.extend(c.output)

  def backward(tensor, depth):
    n = prod.get(tensor)
    if n is None or depth == 0 or n.op_type not in ELEMENTWISE | {'Tanh'}:
      return
    tagged.add(id(n))
    for i in n.input:
      backward(i, depth - 1)

  if coordinates:
    # argmax/top-k indices -> coordinates -> reference points / grids (stops at GridSample etc.)
    forward([o for n in g.node if n.op_type in ('ArgMax', 'TopK') for o in n.output])
    forward(['new_xy'])          # query sampling grid
    backward('points', 4)        # final p_patch + offset

  count = 0
  for n in g.node:
    if id(n) in tagged:
      n.name = TAG + (n.name or f'{n.op_type}_{count}')
      count += 1
  return count


def build(trt, onnx_path, engine_path, precision, profile_shapes, workspace_gb, verbose):
  logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
  model = onnx.load(onnx_path)
  if precision == 'mixed':
    print(f'  tagged {tag_fp32_nodes(model, coordinates="tracker" in os.path.basename(onnx_path))} '
          'ONNX nodes to stay fp32')
  builder = trt.Builder(logger)
  flags = 0
  if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  network = builder.create_network(flags)
  parser = trt.OnnxParser(network, logger)
  if not parser.parse(model.SerializeToString()):
    for i in range(parser.num_errors):
      print(parser.get_error(i))
    sys.exit(f'ONNX parse failed: {onnx_path}')
  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
  if precision in ('mixed', 'fp16'):
    config.set_flag(trt.BuilderFlag.FP16)
  if precision == 'mixed':
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    floats = (trt.float32, trt.float16)
    pinned = 0
    import warnings  # pylint: disable=g-import-not-at-top
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', DeprecationWarning)
      for i in range(network.num_layers):
        layer = network.get_layer(i)
        ident = layer.name + ' ' + (getattr(layer, 'metadata', '') or '')
        if TAG not in ident or layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
          continue
        if all(layer.get_output(k).dtype not in floats for k in range(layer.num_outputs)):
          continue
        layer.precision = trt.float32
        for k in range(layer.num_outputs):
          if layer.get_output(k).dtype in floats:
            layer.set_output_type(k, trt.float32)
        pinned += 1
    print(f'  pinned {pinned}/{network.num_layers} layers to fp32')
  if profile_shapes:
    prof = builder.create_optimization_profile()
    for name, (mn, op, mx) in profile_shapes.items():
      prof.set_shape(name, mn, op, mx)
    config.add_optimization_profile(prof)
  print(f'  building {os.path.basename(engine_path)} ({precision}) ... this takes a few minutes')
  blob = builder.build_serialized_network(network, config)
  if blob is None:
    sys.exit('engine build failed (see TensorRT messages above)')
  with open(engine_path, 'wb') as f:
    f.write(blob)
  print(f'  saved {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)')


def verify(models_dir, enc_engine, trk_engine, video=None, frames=8):
  from export_onnx import test_frames  # pylint: disable=g-import-not-at-top
  clip = [f[0].permute(1, 2, 0).numpy() for f in test_frames(frames, video=video)]
  trt_b = make_backend('trt', models_dir, encoder_engine=enc_engine, tracker_engine=trk_engine)
  try:
    ref_b = make_backend('ort-cuda', models_dir)
  except Exception as e:  # pylint: disable=broad-except
    print(f'  (onnxruntime CUDA unavailable: {e}; using CPU reference)')
    ref_b = make_backend('ort-cpu', models_dir)
  print(f'verifying {trt_b.name} against {ref_b.name}')
  rng = np.random.default_rng(0)
  rand = lambda k: (rng.uniform(0.05, 0.95, (k, 2)) * [512, 384]).astype(np.float32)
  sched = {0: rand(24), 3: rand(16)}
  worst, nan_trt, nan_ref, times = 0.0, 0, 0, []
  for t, f in enumerate(clip):
    if t == 5:  # remove every third point
      for b in (trt_b, ref_b):
        b.remove(range(0, b.n, 3))
    new = sched.get(t, np.zeros((0, 2), np.float32))
    pa, va, _ = trt_b.step(f, new)
    pb, vb, _ = ref_b.step(f, new)
    nan_trt += int(np.isnan(pa).sum() + np.isnan(va).sum())
    nan_ref += int(np.isnan(pb).sum() + np.isnan(vb).sum())
    dxy = float(np.nanmax(np.abs(pa - pb)))
    dv = float(np.nanmax(np.abs(va - vb)))
    worst = max(worst, dxy)
    times.append(trt_b.gpu_ms)
    print(f'  frame {t}: N={trt_b.n:3d}  max|dxy|={dxy:.3f}px  max|dvis|={dv:.4f}  '
          f'NaNs engine {nan_trt} ref {nan_ref}  GPU encoder {trt_b.gpu_ms[0]:.1f} ms tracker {trt_b.gpu_ms[1]:.1f} ms')
  if nan_trt or nan_ref:
    print('FAILED: NaN outputs. Rebuild with --precision fp32 to tell a precision problem from a setup problem.')
    sys.exit(1)
  e, k = np.median([x[0] for x in times[2:]]), np.median([x[1] for x in times[2:]])
  print(f'WORST position difference {worst:.3f}px (model input 512x384 pixels)')
  print(f'median GPU time: encoder {e:.1f} ms + tracker {k:.1f} ms (N~30)')
  print('OK' if worst < 1.0 else 'WARNING: engine deviates by >= 1 px from onnxruntime')


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--models-dir', default=os.path.join(ROOT, 'models'))
  ap.add_argument('--precision', choices=['mixed', 'fp16', 'fp32'], default='mixed')
  ap.add_argument('--encoder-precision', choices=['mixed', 'fp16', 'fp32'])
  ap.add_argument('--tracker-precision', choices=['mixed', 'fp16', 'fp32'])
  ap.add_argument('--min-points', type=int, default=1)
  ap.add_argument('--opt-points', type=int, default=64)
  ap.add_argument('--max-points', type=int, default=1024)
  ap.add_argument('--workspace-gb', type=float, default=2)
  ap.add_argument('--only', choices=['encoder', 'tracker'], help='build just one engine')
  ap.add_argument('--video', help='video for the verification clip (default: synthetic)')
  ap.add_argument('--verify-only', action='store_true')
  ap.add_argument('--no-verify', action='store_true')
  ap.add_argument('--verbose', action='store_true')
  args = ap.parse_args()

  enc_onnx = os.path.join(args.models_dir, 'trackon_encoder.onnx')
  trk_onnx = os.path.join(args.models_dir, 'trackon_tracker.onnx')
  enc_engine = os.path.join(args.models_dir, 'trackon_encoder.engine')
  trk_engine = os.path.join(args.models_dir, 'trackon_tracker.engine')

  if not args.verify_only:
    trt = import_tensorrt()
    print('TensorRT', trt.__version__)
    if args.only != 'tracker':
      build(trt, enc_onnx, enc_engine, args.encoder_precision or args.precision, None,
            args.workspace_gb, args.verbose)
    if args.only != 'encoder':
      tm = onnx.load(trk_onnx, load_external_data=False)
      dims = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in tm.graph.input}
      d, m = dims['q_init'][1], dims['memory'][1]
      prof = {}
      for name, rest in (('q_init', (d,)), ('memory', (m, d)), ('memory_mask', (m,)),
                         ('new_xy', (2,)), ('is_new', ())):
        prof[name] = tuple((n,) + rest for n in (args.min_points, args.opt_points, args.max_points))
      for name in ('f4', 'f8', 'f16', 'f32', 'f_fused'):
        s = tuple(dims[name])
        prof[name] = (s, s, s)
      build(trt, trk_onnx, trk_engine, args.tracker_precision or args.precision, prof,
            args.workspace_gb, args.verbose)
  if not args.no_verify:
    verify(args.models_dir, enc_engine, trk_engine, args.video)


if __name__ == '__main__':
  main()
