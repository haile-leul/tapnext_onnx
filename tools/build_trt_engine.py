"""Build a TensorRT engine (fp16 with fp32 islands) from the TAPNext step ONNX.

TensorRT engines are tied to the GPU architecture and TensorRT version, so
build the engine on (or for) the machine that will run it:

  python tools/build_trt_engine.py \
      --onnx models/tapnext_step_256_w16.onnx \
      --engine models/tapnext_step_256_fp16.engine \
      --min-q 1 --opt-q 64 --max-q 256

The engine works from C++ (same tensor names as the ONNX) and from
run_video.py --backend trt.

Precision modes (--precision):
  mixed (default)  fp16 everywhere except the layers that are numerically
                   sensitive in fp16, which are pinned to fp32: RMSNorm
                   reductions (x^2 can overflow fp16), LayerNorms, the RG-LRU
                   decay / state update (a ~ 1 - 1e-4 is not representable in
                   fp16), and the output heads (coordinates need < 1/8 px).
                   These are all cheap elementwise / per-point ops.
  fp16             TensorRT chooses freely (fastest, may drift).
  fp32             TF32/fp32 reference engine.

After building, --verify (default on) runs the engine and ONNX Runtime on a
synthetic clip and prints the max position / visibility difference.
"""

import argparse
import collections
import os
import sys

import numpy as np
import onnx

NUM_LAYERS, NUM_PATCHES, WIDTH, IMG = 12, 1024, 768, 256
TAG = 'fp32__'
ELEMENTWISE = {'Mul', 'Add', 'Sub', 'Div', 'Sqrt', 'Reciprocal', 'Pow', 'Where',
               'Exp', 'Neg', 'Max', 'Min'}


def einsum_to_matmul(model):
  """Rewrites the graph's Einsum nodes as (Transpose +) MatMul.

  TensorRT's ONNX parser rejects upper-case Einsum subscripts, and MatMul is
  the better-supported path anyway. Exact same maths. Returns count.
  """
  from onnx import helper  # pylint: disable=g-import-not-at-top
  out_nodes, count = [], 0
  for n in model.graph.node:
    eq = next((a.s.decode() for a in n.attribute if a.name == 'equation'), None)
    if n.op_type != 'Einsum':
      out_nodes.append(n)
    elif eq == 'nd,cdD->cnD':  # [n,d] @ [c,d,D] -> [c,n,D]  (MatMul broadcasts)
      out_nodes.append(helper.make_node('MatMul', list(n.input), list(n.output), name=n.name))
      count += 1
    elif eq == 'nhi,hij->nhj':  # block-diagonal linear
      xt, yt = n.output[0] + '__xt', n.output[0] + '__yt'
      out_nodes += [
          helper.make_node('Transpose', [n.input[0]], [xt], perm=[1, 0, 2], name=n.name + '__t0'),
          helper.make_node('MatMul', [xt, n.input[1]], [yt], name=n.name),
          helper.make_node('Transpose', [yt], [n.output[0]], perm=[1, 0, 2], name=n.name + '__t1'),
      ]
      count += 1
    else:  # unknown pattern: at least make the subscripts lower-case
      letters = sorted({c for c in eq if c.isalpha()})
      free = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in letters]
      remap = {c: (c if c.islower() else free.pop(0)) for c in letters}
      for a in n.attribute:
        if a.name == 'equation':
          a.s = ''.join(remap.get(c, c) for c in eq).encode()
      out_nodes.append(n)
  del model.graph.node[:]
  model.graph.node.extend(out_nodes)
  return count


def tag_fp32_nodes(model):
  """Renames ONNX nodes that should stay fp32 with the TAG prefix. Returns count."""
  g = model.graph
  consumers = collections.defaultdict(list)
  producer = {}
  for n in g.node:
    for i in n.input:
      consumers[i].append(n)
    for o in n.output:
      producer[o] = n

  tagged = set()

  def forward(start_nodes, hops, ops):
    frontier = list(start_nodes)
    for _ in range(hops):
      nxt = []
      for n in frontier:
        for o in n.output:
          for c in consumers[o]:
            if c.op_type in ops and id(c) not in tagged:
              tagged.add(id(c))
              nxt.append(c)
      frontier = nxt

  for n in g.node:
    if n.op_type == 'LayerNormalization':
      tagged.add(id(n))
    elif n.op_type == 'ReduceMean':  # RMSNorm: x^2 -> mean -> +eps -> rsqrt
      tagged.add(id(n))
      p = producer.get(n.input[0])
      if p is not None and p.op_type in ('Pow', 'Mul'):
        tagged.add(id(p))
      forward([n], 3, {'Add', 'Sqrt', 'Reciprocal', 'Div'})
    elif n.op_type == 'Exp':  # RG-LRU: a, a^2, sqrt(1-a^2), a*h + x
      tagged.add(id(n))
      forward([n], 6, ELEMENTWISE)

  # Recurrent state path: everything elementwise right after the state inputs.
  state_in = [c for name in ('lru_state', 'conv_state')
              for c in consumers[name]]
  for c in state_in:
    tagged.add(id(c))
  forward(state_in, 4, ELEMENTWISE | {'Gather', 'Slice', 'Concat', 'ReduceSum',
                                      'Unsqueeze', 'Squeeze', 'Reshape'})

  # Output heads + soft-argmax + certainty: all descendants of the encoder
  # norm, i.e. the last LayerNormalization that feeds BOTH heads (the heads
  # have LayerNorms of their own that come later in the graph).
  def descendants(start):
    stack, seen = [start], {}
    while stack:
      n = stack.pop()
      if id(n) in seen:
        continue
      seen[id(n)] = n
      for o in n.output:
        stack.extend(consumers[o])
    return seen

  head_outputs = {id(producer[o]) for o in ('tracks_xy', 'visible_prob')}
  for ln in reversed([n for n in g.node if n.op_type == 'LayerNormalization']):
    desc = descendants(ln)
    if head_outputs <= desc.keys():
      tagged.update(desc.keys())
      break

  count = 0
  for n in g.node:
    if id(n) in tagged:
      n.name = TAG + (n.name or f'{n.op_type}_{count}')
      count += 1
  return count


def build(args):
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from tapnext_runtime import import_tensorrt  # pylint: disable=g-import-not-at-top
  trt = import_tensorrt()
  print('TensorRT', trt.__version__)
  logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING)
  model = onnx.load(args.onnx)
  print('rewrote', einsum_to_matmul(model), 'Einsum nodes as MatMul')
  if args.precision == 'mixed':
    print('tagged', tag_fp32_nodes(model), 'ONNX nodes as fp32')

  builder = trt.Builder(logger)
  flags = 0
  if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  network = builder.create_network(flags)
  parser = trt.OnnxParser(network, logger)
  if not parser.parse(model.SerializeToString()):
    for i in range(parser.num_errors):
      print(parser.get_error(i))
    sys.exit('ONNX parse failed')

  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
  if args.precision in ('mixed', 'fp16'):
    config.set_flag(trt.BuilderFlag.FP16)
  if args.precision == 'mixed':
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    pinned = 0
    floats = (trt.float32, trt.float16)
    for i in range(network.num_layers):
      layer = network.get_layer(i)
      ident = layer.name + ' ' + (getattr(layer, 'metadata', '') or '')
      if TAG not in ident:
        continue
      if layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
        continue
      if all(layer.get_output(k).dtype not in floats for k in range(layer.num_outputs)):
        continue  # index / bool math (ArgMax indices, masks): leave alone
      layer.precision = trt.float32
      for k in range(layer.num_outputs):
        if layer.get_output(k).dtype in floats:
          layer.set_output_type(k, trt.float32)
      pinned += 1
    print(f'pinned {pinned}/{network.num_layers} TensorRT layers to fp32')

  q = (args.min_q, args.opt_q, args.max_q)
  profile = builder.create_optimization_profile()
  for name in ('frame',):
    profile.set_shape(name, (1, IMG, IMG, 3), (1, IMG, IMG, 3), (1, IMG, IMG, 3))
  for name in ('step', 'video_reset'):
    profile.set_shape(name, (1,), (1,), (1,))
  profile.set_shape('query_points', *[(1, x, 3) for x in q])
  profile.set_shape('point_reset', *[(1, x) for x in q])
  profile.set_shape('lru_state', *[(NUM_LAYERS, NUM_PATCHES + x, WIDTH) for x in q])
  profile.set_shape('conv_state', *[(NUM_LAYERS, NUM_PATCHES + x, 3, WIDTH) for x in q])
  config.add_optimization_profile(profile)

  print('building engine (this takes a few minutes)...')
  serialized = builder.build_serialized_network(network, config)
  if serialized is None:
    sys.exit('engine build failed')
  with open(args.engine, 'wb') as f:
    f.write(serialized)
  print('saved', args.engine, os.path.getsize(args.engine) / 1e6, 'MB')


def verify(args):
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from tapnext_runtime import make_backend  # pylint: disable=g-import-not-at-top
  from tools.make_test_video import make_video  # pylint: disable=g-import-not-at-top

  video, tl = make_video()
  q = min(max(16, args.min_q), args.max_q)
  rng = np.random.default_rng(0)
  qp = np.zeros((1, q, 3), np.float32)
  qp[0, :, 1:] = rng.uniform(5, 250, (q, 2))
  qp[0, q // 2:, 0] = 4  # half of the points are queried mid-clip
  trt_b = make_backend('trt', args.engine, q)
  try:
    ref_b = make_backend('ort-cuda', args.onnx, q)
  except Exception:  # pylint: disable=broad-except
    ref_b = make_backend('ort-cpu', args.onnx, q)
  print('reference backend:', ref_b.name)
  worst_xy = worst_vis = 0.0
  for t in range(video.shape[0]):
    ins = (video[t:t + 1], qp, t, float(t == 0), np.zeros((1, q), np.float32))
    a = trt_b.step(*ins)
    b = ref_b.step(*ins)
    active = qp[0, :, 0] <= t
    dxy = np.abs(a[0] - b[0])[active].max()
    dv = np.abs(a[1] - b[1])[active].max()
    worst_xy, worst_vis = max(worst_xy, dxy), max(worst_vis, dv)
    print(f'frame {t}: max|dxy|={dxy:.3f}px  max|dvis|={dv:.4f}')
  print(f'WORST: {worst_xy:.3f} px (256-px coords), visibility {worst_vis:.4f}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--onnx', default='models/tapnext_step_256_w16.onnx')
  ap.add_argument('--engine', default='models/tapnext_step_256_fp16.engine')
  ap.add_argument('--precision', choices=['mixed', 'fp16', 'fp32'], default='mixed')
  ap.add_argument('--min-q', type=int, default=1)
  ap.add_argument('--opt-q', type=int, default=64)
  ap.add_argument('--max-q', type=int, default=256)
  ap.add_argument('--workspace-gb', type=int, default=4)
  ap.add_argument('--no-verify', action='store_true')
  ap.add_argument('--verify-only', action='store_true')
  ap.add_argument('--verbose', action='store_true')
  args = ap.parse_args()
  if not args.verify_only:
    build(args)
  if not args.no_verify:
    verify(args)


if __name__ == '__main__':
  main()
