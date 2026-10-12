"""Run TAPNext (ONNX or TensorRT) on a video and render the point tracks.

Examples
  # GPU via onnxruntime-gpu (CUDA), 8x8 grid on frame 0
  python run_video.py --video gate.mp4 --backend ort-cuda --grid 8

  # native TensorRT engine, lane ROI, re-seed a fresh 6x6 grid every 15 frames
  python run_video.py --video gate.mp4 --backend trt --model models/tapnext_step_256_fp16.engine \
      --roi 400,200,1100,800 --reseed-every 15 --reseed-grid 6

  # add points mid-video: two points at frame 40, a 4x4 grid in a box at frame 55
  python run_video.py --video gate.mp4 --backend ort-cuda \
      --add "40:812,430;900,455" --add-grid "55:700,350,300,200:4"

  # click points yourself on frames 0 and 30 (needs a display)
  python run_video.py --video gate.mp4 --backend ort-cuda --pick 0 --pick 30

  # speed only (no drawing / writing)
  python run_video.py --video gate.mp4 --backend trt --model ... --benchmark

All coordinates are in ORIGINAL video pixels. The ROI is what the model sees
(resized to 256x256); a tight ROI around the lanes = higher effective resolution.
"""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from tapnext_runtime import IMG, PointTracker, make_backend


def parse_box(s):
  x, y, w, h = [float(v) for v in s.split(',')]
  return x, y, w, h


def grid_points(box, n):
  x, y, w, h = box
  xs = x + (np.arange(n) + 0.5) * w / n
  ys = y + (np.arange(n) + 0.5) * h / n
  return [(float(a), float(b)) for b in ys for a in xs]


def build_schedule(args, roi):
  """frame -> list of (x, y) to add at that frame."""
  sched = {}

  def put(f, pts):
    sched.setdefault(int(f), []).extend(pts)

  if args.grid:
    put(0, grid_points(parse_box(args.grid_box) if args.grid_box else roi, args.grid))
  for spec in args.add:
    f, pts = spec.split(':', 1)
    put(f, [tuple(float(v) for v in p.split(',')) for p in pts.split(';') if p])
  for spec in args.add_grid:
    f, box, n = spec.split(':')
    put(f, grid_points(parse_box(box), int(n)))
  if args.points_json:
    with open(args.points_json) as fh:
      for item in json.load(fh):
        if 'points' in item:
          put(item['frame'], [tuple(p) for p in item['points']])
        if 'grid' in item:
          put(item['frame'], grid_points(item['grid'][:4], int(item['grid'][4])))
  return sched


def pick_points(frame, frame_idx):
  """Interactive: left-click to add points, Enter/Space to continue, 'c' to clear."""
  pts, win = [], f'pick points (frame {frame_idx}) - click, Enter=done, c=clear'
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  cv2.setMouseCallback(win, lambda e, x, y, *_: pts.append((float(x), float(y)))
                       if e == cv2.EVENT_LBUTTONDOWN else None)
  while True:
    vis = frame.copy()
    for p in pts:
      cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
    cv2.imshow(win, vis)
    k = cv2.waitKey(20) & 0xFF
    if k in (13, 10, 32):
      break
    if k == ord('c'):
      pts.clear()
  cv2.destroyWindow(win)
  return pts


def color_for(uid):
  hue = (uid * 47) % 180
  c = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
  return int(c[0]), int(c[1]), int(c[2])


def preprocess(frame_bgr, roi):
  x, y, w, h = [int(round(v)) for v in roi]
  crop = frame_bgr[y:y + h, x:x + w]
  rs = cv2.resize(crop, (IMG, IMG), interpolation=cv2.INTER_AREA)
  return cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) * (2.0 / 255.0) - 1.0


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--video', required=True)
  ap.add_argument('--out', default=None, help='output video (default: <video>_tapnext.mp4)')
  ap.add_argument('--backend', default='ort-cuda', choices=['ort-cpu', 'ort-cuda', 'ort-trt', 'trt', 'torch'])
  ap.add_argument('--model', default='models/tapnext_step_256_w16.onnx',
                  help='.onnx for ort-* backends, .engine for trt, bootstapnext_ckpt.npz for torch')
  ap.add_argument('--tapnet-root', default=None, help='tapnet repo clone (only for --backend torch)')
  ap.add_argument('--torch-fp32', action='store_true', help='torch backend without fp16 autocast')
  ap.add_argument('--progress-every', type=int, default=25, help='print a progress line every N frames')
  ap.add_argument('--slots', type=int, default=64, help='point slots Q (fixed per run)')
  ap.add_argument('--roi', default=None, help='x,y,w,h fed to the model (default: full frame)')
  ap.add_argument('--grid', type=int, default=0, help='NxN grid of points at frame 0')
  ap.add_argument('--grid-box', default=None, help='x,y,w,h for --grid (default: ROI)')
  ap.add_argument('--add', action='append', default=[], help='"F:x,y;x,y;..." add points at frame F')
  ap.add_argument('--add-grid', action='append', default=[], help='"F:x,y,w,h:N" NxN grid in box at frame F')
  ap.add_argument('--points-json', default=None,
                  help='[{"frame":F,"points":[[x,y],...]}, {"frame":F,"grid":[x,y,w,h,N]}]')
  ap.add_argument('--pick', action='append', type=int, default=[], help='click points on frame F')
  ap.add_argument('--reseed-every', type=int, default=0, help='add a fresh grid every K frames')
  ap.add_argument('--reseed-grid', type=int, default=6)
  ap.add_argument('--reseed-box', default=None, help='x,y,w,h for re-seeding (default: ROI)')
  ap.add_argument('--drop-after', type=int, default=0,
                  help='free a slot after N consecutive unreliable frames (0 = never)')
  ap.add_argument('--start', type=int, default=0, help='first video frame to use')
  ap.add_argument('--max-frames', type=int, default=0)
  ap.add_argument('--trail', type=int, default=20, help='trail length in frames')
  ap.add_argument('--vel-scale', type=float, default=1.0, help='velocity arrow length multiplier')
  ap.add_argument('--show-unreliable', action='store_true', help='also draw unreliable points (hollow)')
  ap.add_argument('--csv', default=None, help='write per-point tracks to CSV')
  ap.add_argument('--show', action='store_true', help='live window')
  ap.add_argument('--benchmark', action='store_true', help='no drawing/writing, timing only')
  ap.add_argument('--warmup', type=int, default=5, help='frames excluded from timing stats')
  ap.add_argument('--trt-cache', default=None, help='engine cache dir for ort-trt')
  args = ap.parse_args()

  cap = cv2.VideoCapture(args.video)
  if not cap.isOpened():
    raise SystemExit(f'cannot open {args.video}')
  fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
  W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  if args.start:
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
  roi = parse_box(args.roi) if args.roi else (0.0, 0.0, float(W), float(H))

  print(f'loading {args.backend}: {args.model}')
  t0 = time.time()
  backend = make_backend(args.backend, args.model, args.slots, trt_cache=args.trt_cache,
                         tapnet_root=args.tapnet_root, torch_fp32=args.torch_fp32)
  print(f'  {backend.name} ready in {time.time() - t0:.1f}s')
  tracker = PointTracker(backend, args.slots, roi)
  sched = build_schedule(args, roi)
  reseed_box = parse_box(args.reseed_box) if args.reseed_box else roi

  writer = None
  if args.benchmark:
    print('benchmark mode: no video is written (drop --benchmark to write one)')
  else:
    out = os.path.abspath(args.out or os.path.splitext(args.video)[0] + '_tapnext.mp4')
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    if not writer.isOpened():  # no mp4 encoder in this OpenCV build: fall back to MJPG .avi
      out = os.path.splitext(out)[0] + '.avi'
      writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))
    if not writer.isOpened():
      raise SystemExit(f'cannot open a video writer for {out}')
    print('output video:', out, '(finalised at the end or on Ctrl+C)')
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - args.start
  if args.max_frames:
    n_frames = min(n_frames, args.max_frames) if n_frames > 0 else args.max_frames
  csv_w = None
  if args.csv:
    csv_f = open(args.csv, 'w', newline='')  # pylint: disable=consider-using-with
    csv_w = csv.writer(csv_f)
    csv_w.writerow(['frame', 'point_id', 'slot', 'x', 'y', 'vx', 'vy', 'visible', 'certainty', 'reliable'])

  trails = {}  # uid -> list of (x, y, reliable)
  model_ms, pre_ms, total_ms, gpu_ms = [], [], [], []
  t = 0
  t_wall = time.perf_counter()
  try:
    while True:
      ok, frame = cap.read()
      if not ok or (args.max_frames and t >= args.max_frames):
        break
      t_start = time.perf_counter()

      # --- schedule points for this frame
      new_pts = list(sched.get(t, []))
      if t in args.pick:
        new_pts += pick_points(frame, t)
      if args.reseed_every and t > 0 and t % args.reseed_every == 0:
        new_pts += grid_points(reseed_box, args.reseed_grid)
      added = 0
      for (x, y) in new_pts:
        added += tracker.add(x, y) >= 0
      if new_pts:
        print(f'frame {t}: added {added}/{len(new_pts)} points '
              f'({int(tracker.active.sum())}/{args.slots} slots active)')

      # --- model
      tp = time.perf_counter()
      inp = preprocess(frame, roi)
      tm = time.perf_counter()
      reliable = tracker.step(inp)
      te = time.perf_counter()
      if args.drop_after:
        for s in np.flatnonzero(tracker.active & (tracker.bad_streak >= args.drop_after)):
          tracker.remove(s)

      # --- output
      act = np.flatnonzero(tracker.active)
      if csv_w:
        for s in act:
          csv_w.writerow([t + args.start, tracker.uid[s], s, *np.round(tracker.xy[s], 2),
                          *np.round(tracker.v[s], 2), round(float(tracker.vis[s]), 3),
                          round(float(tracker.cert[s]), 3), int(reliable[s])])
      if writer is not None or args.show:
        vis = frame
        rx, ry, rw, rh = [int(round(v)) for v in roi]
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)
        live = set()
        for s in act:
          u = int(tracker.uid[s])
          live.add(u)
          tr = trails.setdefault(u, [])
          tr.append((float(tracker.xy[s, 0]), float(tracker.xy[s, 1]), bool(reliable[s])))
          del tr[:-args.trail]
        for u in list(trails):
          if u not in live:
            del trails[u]
        for s in act:
          u = int(tracker.uid[s])
          c = color_for(u)
          tr = trails[u]
          for (x0, y0, r0), (x1, y1, r1) in zip(tr[:-1], tr[1:]):
            if (r0 and r1) or args.show_unreliable:
              cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), c, 2, cv2.LINE_AA)
          x, y = int(tracker.xy[s, 0]), int(tracker.xy[s, 1])
          if reliable[s]:
            cv2.circle(vis, (x, y), 5, c, -1, cv2.LINE_AA)
            v = tracker.v[s] * args.vel_scale
            if np.hypot(*v) > 1:
              cv2.arrowedLine(vis, (x, y), (int(x + v[0]), int(y + v[1])), (255, 255, 255), 2,
                              cv2.LINE_AA, tipLength=0.25)
          elif args.show_unreliable:
            cv2.circle(vis, (x, y), 5, c, 1, cv2.LINE_AA)
        ms = np.mean(model_ms[-30:]) if model_ms else (te - tm) * 1000
        hud = (f'frame {t + args.start}  {backend.name}  model {ms:.1f} ms ({1000 / max(ms, 1e-3):.0f} fps)  '
               f'points {int(reliable[act].sum())}/{len(act)} reliable')
        cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is not None:
          writer.write(vis)
        if args.show:
          cv2.imshow('tapnext', vis)
          if (cv2.waitKey(1) & 0xFF) == 27:
            break

      if t >= args.warmup:
        pre_ms.append((tm - tp) * 1000)
        model_ms.append((te - tm) * 1000)
        if getattr(backend, 'gpu_ms', None) is not None:
          gpu_ms.append(backend.gpu_ms)
        total_ms.append((time.perf_counter() - t_start) * 1000)
      t += 1

      if args.progress_every and t % args.progress_every == 0 and model_ms:
        el = time.perf_counter() - t_wall
        g = f', engine GPU {np.mean(gpu_ms[-args.progress_every:]):.1f} ms' if gpu_ms else ''
        print(f'frame {t}/{n_frames if n_frames > 0 else "?"}: model {np.mean(model_ms[-args.progress_every:]):.1f} ms'
              f'{g}, overall {t / el:.1f} fps', flush=True)
  except KeyboardInterrupt:
    print(f'\ninterrupted at frame {t}: finalising outputs...')

  cap.release()
  if writer is not None:
    writer.release()
    print('wrote', out)
  if csv_w:
    csv_f.close()
    print('wrote', args.csv)
  if model_ms:
    m = np.array(model_ms)
    print(f'\n{backend.name}, Q={args.slots}, {len(m)} timed frames (after {args.warmup} warm-up):')
    print(f'  model   mean {m.mean():.2f} ms  p50 {np.median(m):.2f}  p95 {np.percentile(m, 95):.2f}'
          f'  -> {1000 / m.mean():.1f} fps')
    if gpu_ms:
      g = np.array(gpu_ms)
      print(f'  engine GPU time (CUDA events) mean {g.mean():.2f} ms  p50 {np.median(g):.2f}'
            f'  -> {1000 / g.mean():.1f} fps   (model - GPU = host/PCIe overhead)')
    print(f'  preproc mean {np.mean(pre_ms):.2f} ms   end-to-end (incl. drawing/IO) mean {np.mean(total_ms):.2f} ms')


if __name__ == '__main__':
  main()
