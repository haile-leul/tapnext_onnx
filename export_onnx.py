"""Export Track-On2 / Track-On-R to two ONNX graphs and verify them against PyTorch.

  python export_onnx.py --trackon-root ../track_on --ckpt ../trackon2_dinov3_checkpoint.pt

writes  models/trackon_encoder.onnx   frame -> features   (run once per frame)
        models/trackon_tracker.onnx   features + point state -> points + new state
                                      (N points is dynamic: add / remove points freely)

Then it runs the ONNX models with onnxruntime next to the original PyTorch code
(Track_On2.extract_frame_features + track_frame + the Predictor's memory update)
on a short clip with points added mid-way and prints the differences.

Needs: torch, transformers, timm, onnx, onnxscript, onnxruntime, opencv-python
and HuggingFace access to facebook/dinov3-vits16plus-pretrain-lvd1689m
(`huggingface-cli login`). No mmcv required.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from trackon_export.loading import build_model  # pylint: disable=g-import-not-at-top
from trackon_export.modules import TrackOnEncoder, TrackOnTracker  # pylint: disable=g-import-not-at-top
from trackon_export.msda import record_shapes  # pylint: disable=g-import-not-at-top

ENC_IN = ['frame']
ENC_OUT = ['f4', 'f8', 'f16', 'f32', 'f_fused']
TRK_IN = ENC_OUT + ['q_init', 'memory', 'memory_mask', 'new_xy', 'is_new']
TRK_OUT = ['points', 'vis_logit', 'unc_logit', 'q_init_out', 'memory_out', 'memory_mask_out']


def test_frames(n, h=384, w=512, seed=0, video=None):
  """n RGB float frames [1,3,h,w] in 0..255: from a video if given, else a moving texture."""
  if video:
    import cv2  # pylint: disable=g-import-not-at-top
    cap, out = cv2.VideoCapture(video), []
    while len(out) < n:
      ok, f = cap.read()
      if not ok:
        break
      f = cv2.cvtColor(cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
      out.append(torch.from_numpy(f).permute(2, 0, 1)[None].float())
    if len(out) == n:
      return out
  g = torch.Generator().manual_seed(seed)
  base = sum(F.interpolate(torch.rand(1, 3, h // s + 2, w // s + 2, generator=g), size=(h + 64, w + 64),
                           mode='bicubic', align_corners=False) for s in (4, 8, 16, 32))
  base = (base - base.min()) / (base.max() - base.min()) * 255
  return [base[:, :, 10 + 3 * t:10 + 3 * t + h, 20 + 7 * t:20 + 7 * t + w].contiguous() for t in range(n)]


def reference_run(model, frames, schedule):
  """Original code path: extract_frame_features + init (grid_sample) + track_frame + roll."""
  m, d, mm = model, model.D, model.M
  q = mem = mask = None
  outs = []
  with torch.no_grad():
    for t, fr in enumerate(frames):
      feats = m.extract_frame_features(fr)
      new = schedule.get(t)
      if new is not None:
        fmap = feats[4].view(1, m.Hf, m.Wf, d).permute(0, 3, 1, 2)
        g = torch.stack([new[:, 0] / m.W, new[:, 1] / m.H], -1) * 2 - 1
        qn = F.grid_sample(fmap, g.view(1, -1, 1, 2), mode='bilinear', padding_mode='border',
                           align_corners=False)[0, :, :, 0].T
        z = lambda *s: torch.zeros(*s)
        q = qn if q is None else torch.cat([q, qn])
        mem = z(len(qn), mm, d) if mem is None else torch.cat([mem, z(len(qn), mm, d)])
        mk = torch.ones(len(qn), mm, dtype=torch.bool)
        mask = mk if mask is None else torch.cat([mask, mk])
      p, v, qnew = m.track_frame(q, mask, mem, feats, m.H, m.W)
      mem = torch.roll(mem, -1, 1)
      mem[:, -1] = qnew
      mask = torch.roll(mask, -1, 1)
      mask[:, -1] = False
      outs.append((p.numpy(), v.numpy()))
  return outs


def onnx_run(enc_path, trk_path, frames, schedule, d, mm, providers):
  import onnxruntime as ort  # pylint: disable=g-import-not-at-top
  enc = ort.InferenceSession(enc_path, providers=providers)
  trk = ort.InferenceSession(trk_path, providers=providers)
  q = np.zeros((0, d), np.float32)
  mem = np.zeros((0, mm, d), np.float32)
  mask = np.zeros((0, mm), np.float32)
  outs, t_enc, t_trk = [], [], []
  for t, fr in enumerate(frames):
    t0 = time.perf_counter()
    feats = enc.run(None, {'frame': fr.numpy()})
    t1 = time.perf_counter()
    new = schedule.get(t)
    n_new = 0 if new is None else len(new)
    is_new = np.concatenate([np.zeros(len(q)), np.ones(n_new)]).astype(np.float32)
    new_xy = np.zeros((len(q) + n_new, 2), np.float32)
    if n_new:
      new_xy[len(q):] = new.numpy()
    q = np.concatenate([q, np.zeros((n_new, d), np.float32)])
    mem = np.concatenate([mem, np.zeros((n_new, mm, d), np.float32)])
    mask = np.concatenate([mask, np.ones((n_new, mm), np.float32)])
    p, v, _, q, mem, mask = trk.run(None, dict(zip(TRK_IN, feats + [q, mem, mask, new_xy, is_new])))
    t2 = time.perf_counter()
    outs.append((p, v))
    t_enc.append(t1 - t0)
    t_trk.append(t2 - t1)
  return outs, 1000 * np.median(t_enc), 1000 * np.median(t_trk)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--trackon-root', required=True, help='clone of github.com/gorkaydemir/track_on')
  ap.add_argument('--ckpt', help='Track-On2 or Track-On-R checkpoint (.pt)')
  ap.add_argument('--config', help='model yaml (default: config/test.yaml values)')
  ap.add_argument('--out-dir', default=os.path.join(ROOT, 'models'))
  ap.add_argument('--max-points', type=int, default=8192, help='upper bound of the dynamic N axis')
  ap.add_argument('--opset', type=int, default=18)
  ap.add_argument('--video', help='optional video for the verification clip')
  ap.add_argument('--fake-dinov3', action='store_true',
                  help='random DINOv3 weights (pipeline test without HuggingFace access)')
  ap.add_argument('--skip-export', action='store_true', help='only re-run the verification')
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  enc_path = os.path.join(args.out_dir, 'trackon_encoder.onnx')
  trk_path = os.path.join(args.out_dir, 'trackon_tracker.onnx')

  model, margs = build_model(args.trackon_root, args.ckpt, args.config, fake_dinov3=args.fake_dinov3)
  h, w = margs.input_size
  d, mm, p_tok = model.D, model.M, model.P
  print(f'model: input {h}x{w}, D={d}, memory M={mm}, feature tokens P={p_tok}')

  frames = test_frames(6, h, w, video=args.video)
  # warm-up eager pass: records static spatial shapes of all deformable-attention modules
  with torch.no_grad(), record_shapes(model):
    feats = model.extract_frame_features(frames[0])
    model.track_frame(torch.zeros(4, d), torch.ones(4, mm, dtype=torch.bool),
                      torch.zeros(4, mm, d), feats, h, w)

  if not args.skip_export:
    enc, trk = TrackOnEncoder(model).eval(), TrackOnTracker(model).eval()
    print('exporting encoder ...')
    with torch.no_grad():
      ep = torch.onnx.export(enc, (frames[0],), input_names=ENC_IN, output_names=ENC_OUT,
                             opset_version=args.opset, dynamo=True, optimize=True)
    ep.save(enc_path)
    print('  saved', enc_path, f'{os.path.getsize(enc_path) / 1e6:.1f} MB')

    n = 8
    with torch.no_grad():
      ef = enc(frames[0])
    ex = (*ef, torch.randn(n, d), torch.zeros(n, mm, d), torch.ones(n, mm),
          torch.rand(n, 2) * torch.tensor([w, h]), torch.ones(n))
    pn = torch.export.Dim('num_points', min=1, max=args.max_points)
    dyn = {k: None for k in ENC_OUT}
    dyn.update(q_init={0: pn}, memory={0: pn}, memory_mask={0: pn}, new_xy={0: pn}, is_new={0: pn})
    print('exporting tracker (dynamic number of points) ...')
    with torch.no_grad():
      tp = torch.onnx.export(trk, ex, input_names=TRK_IN, output_names=TRK_OUT, dynamic_shapes=dyn,
                             opset_version=args.opset, dynamo=True, optimize=True)
    tp.save(trk_path)
    print('  saved', trk_path, f'{os.path.getsize(trk_path) / 1e6:.1f} MB')

  # --- verification: 3 points from frame 0, 2 added at frame 2, 1 at frame 4
  sched = {0: torch.tensor([[100., 100.], [300., 200.], [480., 370.]]),
           2: torch.tensor([[50., 60.], [250., 250.]]), 4: torch.tensor([[400., 50.]])}
  ref = reference_run(model, frames, sched)
  out, te, tt = onnx_run(enc_path, trk_path, frames, sched, d, mm, ['CPUExecutionProvider'])
  worst = 0.0
  for t, ((rp, rv), (op, ov)) in enumerate(zip(ref, out)):
    dp = float(np.abs(rp - op).max())
    worst = max(worst, dp)
    print(f'frame {t}: N={len(op)}  max|dxy|={dp:.2e}px  max|dvis_logit|={np.abs(rv - ov).max():.2e}')
  print(f'onnxruntime CPU: encoder {te:.0f} ms, tracker {tt:.0f} ms per frame')
  print('OK' if worst < 0.05 else f'WARNING: max deviation {worst:.3f}px')


if __name__ == '__main__':
  main()
