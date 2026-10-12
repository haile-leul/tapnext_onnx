"""Prints what the GPU backends need, so problems show up before a long engine build.

  python tools/check_gpu_env.py
"""
import importlib
import platform
import sys

print('python', sys.version.split()[0], platform.system(), platform.machine())


def ver(mod):
  try:
    m = importlib.import_module(mod)
    return m, getattr(m, '__version__', '?')
  except Exception as e:  # pylint: disable=broad-except
    return None, f'NOT AVAILABLE ({type(e).__name__}: {e})'


ok = True
torch, v = ver('torch')
print('torch       ', v)
if torch is not None:
  print('  cuda build:', torch.version.cuda, '| cuda available:', torch.cuda.is_available())
  if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  compute capability {p.major}.{p.minor}, {p.total_memory / 2**30:.1f} GB')
  if torch.cuda.is_available():
    print('  gpu:', torch.cuda.get_device_name(0))
  else:
    ok = False
    print('  -> install a CUDA build of torch (see README, Windows section)')

ort, v = ver('onnxruntime')
print('onnxruntime ', v)
if ort is not None:
  import os  # pylint: disable=g-import-not-at-top
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from tapnext_runtime import _load_cuda_dlls  # pylint: disable=g-import-not-at-top
  _load_cuda_dlls(ort)
  print('  compiled-in providers:', ort.get_available_providers())
  # get_available_providers() only lists what was compiled in; a real session
  # shows whether CUDA actually loads (otherwise ORT silently falls back to CPU).
  import numpy as np  # pylint: disable=g-import-not-at-top
  from onnx import TensorProto, helper  # pylint: disable=g-import-not-at-top
  g = helper.make_graph([helper.make_node('MatMul', ['a', 'a'], ['b'])], 'g',
                        [helper.make_tensor_value_info('a', TensorProto.FLOAT, [64, 64])],
                        [helper.make_tensor_value_info('b', TensorProto.FLOAT, [64, 64])])
  m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
  m.ir_version = 8
  try:
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    sess.run(None, {'a': np.ones((64, 64), np.float32)})
    used = sess.get_providers()[0]
  except Exception as e:  # pylint: disable=broad-except
    used = f'FAILED ({e})'
  print('  test session runs on:', used)
  if used != 'CUDAExecutionProvider':
    ok = False
    print('  -> CUDA EP not usable. onnxruntime-gpu, torch and tensorrt must all use the same CUDA major '
          '(see README, Windows section); also make sure plain `onnxruntime` is NOT installed.')

try:
  import os  # pylint: disable=g-import-not-at-top
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from tapnext_runtime import import_tensorrt  # pylint: disable=g-import-not-at-top
  trt = import_tensorrt()
  v = trt.__version__
except Exception as e:  # pylint: disable=broad-except
  trt, v = None, f'NOT AVAILABLE ({type(e).__name__}: {e})'
print('tensorrt    ', v)
if trt is None:
  ok = False
else:
  try:
    b = trt.Builder(trt.Logger(trt.Logger.WARNING))
    print('  builder OK, fast fp16:', b.platform_has_fast_fp16)
  except Exception as e:  # pylint: disable=broad-except
    ok = False
    print('  builder FAILED:', e)

cv2, v = ver('cv2')
print('opencv      ', v)
onnx, v = ver('onnx')
print('onnx        ', v)
print('\nALL GOOD' if ok and cv2 and onnx else '\nfix the items above first')
