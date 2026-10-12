"""Joins models/*.onnx.partNN into the .onnx and checks its SHA-256 (works on Windows too).

  python tools/reassemble_model.py
"""
import glob
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = os.path.join(ROOT, 'models', 'tapnext_step_256_w16.onnx')
parts = sorted(glob.glob(name + '.part*'))
if not parts:
  sys.exit(f'no parts found next to {name}')
if not os.path.exists(name):
  with open(name, 'wb') as out:
    for p in parts:
      with open(p, 'rb') as f:
        out.write(f.read())
  print(f'joined {len(parts)} parts -> {name}')
h = hashlib.sha256()
with open(name, 'rb') as f:
  for chunk in iter(lambda: f.read(1 << 24), b''):
    h.update(chunk)
expected = open(name + '.sha256').read().split()[0]
if h.hexdigest() != expected:
  sys.exit(f'CHECKSUM MISMATCH: delete {name} and re-run (got {h.hexdigest()})')
print('checksum OK:', name)
