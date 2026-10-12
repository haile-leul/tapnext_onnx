"""Run Track-On2 on a video (PyTorch, ONNX Runtime or TensorRT) and render the tracks.

Examples (all coordinates in ORIGINAL video pixels):
  # TensorRT, 8x8 grid on the first frame, live window
  python run_video.py --video gate.mp4 --backend trt --grid 8 --show

  # add points mid-video, remove some later
  python run_video.py --video gate.mp4 --backend trt \
      --add "40:812,430;900,455" --add-grid "55:700,350,300,200:5" \
      --remove "80:0,1" --remove-box "120:0,0,400,1080"

  # fresh 6x6 grid every 15 frames, drop points invisible for 10 frames
  python run_video.py --video gate.mp4 --backend trt --reseed-every 15 --reseed-grid 6 --drop-after 10

  # original PyTorch model / ONNX Runtime
  python run_video.py --video gate.mp4 --backend torch --trackon-root ../track_on --ckpt ../trackon2_dinov3_checkpoint.pt
  python run_video.py --video gate.mp4 --backend ort-cuda

Output video next to the input (<name>_trackon.mp4), finalised at the end or on
Ctrl+C. Timing summary at the end (TensorRT also reports pure GPU time).
"""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from trackon_runtime import PointTracker, make_backend


def box(s):
  return tuple(float(v) for v in s.split(','))


def grid(b, n):
  x, y, w, h = b
  return [(x + (i + 0.5) * w / n, y + (j + 0.5) * h / n) for j in range(n) for i in range(n)]


def build_schedule(args, full):
  add, rem, rem_box = {}, {}, {}
  if args.grid:
    add.setdefault(0, []).extend(grid(box(args.grid_box) if args.grid_box else full, args.grid))
  for s in args.add:
    f, pts = s.split(':', 1)
    add.setdefault(int(f), []).extend(tuple(float(v) for v in p.split(',')) for p in pts.split(';') if p)
  for s in args.add_grid:
    f, b, n = s.split(':')
    add.setdefault(int(f), []).extend(grid(box(b), int(n)))
  for s in args.remove:
    f, ids = s.split(':')
    rem.setdefault(int(f), []).extend(int(i) for i in ids.split(',') if i)
  for s in args.remove_box:
    f, b = s.split(':')
    rem_box.setdefault(int(f), []).append(box(b))
  if args.points_json:
    for item in json.load(open(args.points_json)):
      f = int(item['frame'])
      add.setdefault(f, []).extend(tuple(p) for p in item.get('points', []))
      if 'grid' in item:
        add.setdefault(f, []).extend(grid(item['grid'][:4], int(item['grid'][4])))
      rem.setdefault(f, []).extend(item.get('remove', []))
  return add, rem, rem_box


def pick(frame, t):
  pts, win = [], f'frame {t}: click points, Enter = done, c = clear'
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  cv2.setMouseCallback(win, lambda e, x, y, *_: pts.append((float(x), float(y))) if e == cv2.EVENT_LBUTTONDOWN else None)
  while True:
    v = frame.copy()
    for p in pts:
      cv2.circle(v, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
    cv2.imshow(win, v)
    k = cv2.waitKey(20) & 0xFF
    if k in (10, 13, 32):
      break
    if k == ord('c'):
      pts.clear()
  cv2.destroyWindow(win)
  return pts


def color(i):
  c = cv2.cvtColor(np.uint8([[[(i * 47) % 180, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
  return int(c[0]), int(c[1]), int(c[2])


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--video', required=True)
  ap.add_argument('--backend', default='trt', choices=['trt', 'ort-cuda', 'ort-cpu', 'torch'])
  ap.add_argument('--models-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
  ap.add_argument('--trackon-root', help='track_on repo clone (torch backend)')
  ap.add_argument('--ckpt', help='Track-On2 / Track-On-R checkpoint (torch backend)')
  ap.add_argument('--torch-fp16', action='store_true', help='torch backend with fp16 autocast')
  ap.add_argument('--roi', help='x,y,w,h region given to the model (default: full frame)')
  ap.add_argument('--grid', type=int, default=0, help='NxN grid at frame 0')
  ap.add_argument('--grid-box', help='x,y,w,h for --grid (default: ROI / full frame)')
  ap.add_argument('--add', action='append', default=[], help='"F:x,y;x,y" add points at frame F')
  ap.add_argument('--add-grid', action='append', default=[], help='"F:x,y,w,h:N" NxN grid in a box at frame F')
  ap.add_argument('--remove', action='append', default=[], help='"F:id,id" remove point ids at frame F')
  ap.add_argument('--remove-box', action='append', default=[], help='"F:x,y,w,h" remove points inside a box at frame F')
  ap.add_argument('--points-json', help='[{"frame":F,"points":[[x,y]],"grid":[x,y,w,h,N],"remove":[ids]}]')
  ap.add_argument('--pick', action='append', type=int, default=[], help='click points on frame F (needs a display)')
  ap.add_argument('--reseed-every', type=int, default=0, help='add a fresh grid every K frames')
  ap.add_argument('--reseed-grid', type=int, default=6)
  ap.add_argument('--reseed-box', help='x,y,w,h for re-seeding (default: ROI / full frame)')
  ap.add_argument('--drop-after', type=int, default=0, help='remove points invisible for N frames in a row')
  ap.add_argument('--max-points', type=int, default=0, help='cap: drop the oldest points beyond this')
  ap.add_argument('--start', type=int, default=0)
  ap.add_argument('--max-frames', type=int, default=0)
  ap.add_argument('--trail', type=int, default=20)
  ap.add_argument('--show-invisible', action='store_true')
  ap.add_argument('--out', help='output video (default <video>_trackon.mp4)')
  ap.add_argument('--csv', help='write per-point tracks')
  ap.add_argument('--show', action='store_true', help='live window (Esc stops)')
  ap.add_argument('--benchmark', action='store_true', help='no drawing / video writing')
  ap.add_argument('--warmup', type=int, default=5)
  ap.add_argument('--progress-every', type=int, default=25)
  ap.add_argument('--fake-dinov3', action='store_true', help=argparse.SUPPRESS)
  args = ap.parse_args()

  cap = cv2.VideoCapture(args.video)
  if not cap.isOpened():
    raise SystemExit(f'cannot open {args.video}')
  fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
  W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - args.start
  if args.max_frames:
    n_frames = min(n_frames, args.max_frames) if n_frames > 0 else args.max_frames
  if args.start:
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
  roi = box(args.roi) if args.roi else None
  full = roi or (0.0, 0.0, float(W), float(H))

  t0 = time.time()
  backend = make_backend(args.backend, models_dir=args.models_dir, trackon_root=args.trackon_root,
                         ckpt=args.ckpt, fp16=args.torch_fp16, fake_dinov3=args.fake_dinov3)
  print(f'{backend.name} ready in {time.time() - t0:.1f}s')
  tracker = PointTracker(backend, roi=roi)
  add, rem, rem_box = build_schedule(args, full)
  reseed_box = box(args.reseed_box) if args.reseed_box else full

  writer, out = None, None
  if not args.benchmark:
    out = os.path.abspath(args.out or os.path.splitext(args.video)[0] + '_trackon.mp4')
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    if not writer.isOpened():
      out = os.path.splitext(out)[0] + '.avi'
      writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))
    print('output video:', out)
  else:
    print('benchmark mode: no video written')
  csv_f = open(args.csv, 'w', newline='') if args.csv else None  # pylint: disable=consider-using-with
  csv_w = csv.writer(csv_f) if csv_f else None
  if csv_w:
    csv_w.writerow(['frame', 'id', 'x', 'y', 'vx', 'vy', 'vis_prob', 'visible'])

  trails, step_ms, gpu = {}, [], []
  t, t_wall = 0, time.perf_counter()
  try:
    while True:
      ok, frame = cap.read()
      if not ok or (args.max_frames and t >= args.max_frames):
        break
      # ---- point edits for this frame
      ids_rm = list(rem.get(t, []))
      for b in rem_box.get(t, []):
        ids_rm += [i for i, s in tracker.state.items()
                   if b[0] <= s['x'] < b[0] + b[2] and b[1] <= s['y'] < b[1] + b[3]]
      if args.drop_after:
        ids_rm += [i for i, s in tracker.state.items() if s['lost'] >= args.drop_after]
      new = list(add.get(t, []))
      if t in args.pick:
        new += pick(frame, t)
      if args.reseed_every and t > 0 and t % args.reseed_every == 0:
        new += grid(reseed_box, args.reseed_grid)
      if args.max_points and tracker.num_points + len(new) - len(set(ids_rm)) > args.max_points:
        over = tracker.num_points + len(new) - len(set(ids_rm)) - args.max_points
        ids_rm += [i for i in tracker.ids if i not in ids_rm][:over]   # oldest first
      if ids_rm:
        tracker.remove(ids_rm)
      for x, y in new:
        tracker.add(x, y)
      if ids_rm or new:
        print(f'frame {t}: +{len(new)} -{len(set(ids_rm))} -> {tracker.num_points} points')

      # ---- model
      ts = time.perf_counter()
      res = tracker.step(frame)
      te = time.perf_counter()
      if t >= args.warmup and res:
        step_ms.append((te - ts) * 1000)
        if backend.gpu_ms:
          gpu.append(backend.gpu_ms)

      if csv_w:
        for i, s in res.items():
          csv_w.writerow([t + args.start, i, f"{s['x']:.2f}", f"{s['y']:.2f}", f"{s['vx']:.2f}",
                          f"{s['vy']:.2f}", f"{s['vis']:.3f}", int(s['visible'])])
      if writer is not None or args.show:
        vis = frame
        if roi:
          cv2.rectangle(vis, (int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (255, 255, 255), 1)
        for i in list(trails):
          if i not in res:
            del trails[i]
        for i, s in res.items():
          tr = trails.setdefault(i, [])
          tr.append((int(s['x']), int(s['y']), s['visible']))
          del tr[:-args.trail]
          c = color(i)
          for (x0, y0, v0), (x1, y1, v1) in zip(tr[:-1], tr[1:]):
            if (v0 and v1) or args.show_invisible:
              cv2.line(vis, (x0, y0), (x1, y1), c, 2, cv2.LINE_AA)
          if s['visible']:
            cv2.circle(vis, (int(s['x']), int(s['y'])), 5, c, -1, cv2.LINE_AA)
            if abs(s['vx']) + abs(s['vy']) > 1:
              cv2.arrowedLine(vis, (int(s['x']), int(s['y'])), (int(s['x'] + s['vx']), int(s['y'] + s['vy'])),
                              (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
          elif args.show_invisible:
            cv2.circle(vis, (int(s['x']), int(s['y'])), 5, c, 1, cv2.LINE_AA)
        ms = np.mean(step_ms[-30:]) if step_ms else (te - ts) * 1000
        n_vis = sum(s['visible'] for s in res.values())
        hud = f'frame {t + args.start}  {backend.name}  {ms:.1f} ms ({1000 / max(ms, 1e-3):.1f} fps)  points {n_vis}/{len(res)} visible'
        cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is not None:
          writer.write(vis)
        if args.show:
          cv2.imshow('track-on2', vis)
          if (cv2.waitKey(1) & 0xFF) == 27:
            break
      t += 1
      if args.progress_every and t % args.progress_every == 0 and step_ms:
        g = f', GPU enc {np.mean([x[0] for x in gpu[-25:]]):.1f} + trk {np.mean([x[1] for x in gpu[-25:]]):.1f} ms' if gpu else ''
        print(f'frame {t}/{n_frames if n_frames > 0 else "?"}: step {np.mean(step_ms[-25:]):.1f} ms{g}, '
              f'{tracker.num_points} points, overall {t / (time.perf_counter() - t_wall):.1f} fps', flush=True)
  except KeyboardInterrupt:
    print(f'\ninterrupted at frame {t}: finalising outputs ...')

  cap.release()
  if writer is not None:
    writer.release()
    print('wrote', out)
  if csv_f:
    csv_f.close()
    print('wrote', args.csv)
  if step_ms:
    s = np.array(step_ms)
    print(f'\n{backend.name}: {len(s)} timed frames (after {args.warmup} warm-up)')
    print(f'  step (preprocess + model + copies) mean {s.mean():.1f} ms  p50 {np.median(s):.1f}  '
          f'p95 {np.percentile(s, 95):.1f}  -> {1000 / s.mean():.1f} fps')
    if gpu:
      e, k = np.array([x[0] for x in gpu]), np.array([x[1] for x in gpu])
      print(f'  GPU only: encoder {e.mean():.1f} ms + tracker {k.mean():.1f} ms = {(e + k).mean():.1f} ms '
            f'-> {1000 / (e + k).mean():.1f} fps')


if __name__ == '__main__':
  main()
