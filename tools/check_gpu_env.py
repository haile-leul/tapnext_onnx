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
    print('  gpu:', torch.cuda.get_device_name(0))
  else:
    ok = False
    print('  -> install a CUDA build of torch (see README, Windows section)')

ort, v = ver('onnxruntime')
print('onnxruntime ', v)
if ort is not None:
  if hasattr(ort, 'preload_dlls'):
    try:
      ort.preload_dlls()
    except Exception as e:  # pylint: disable=broad-except
      print('  preload_dlls:', e)
  provs = ort.get_available_providers()
  print('  providers:', provs)
  if 'CUDAExecutionProvider' not in provs:
    ok = False
    print('  -> pip uninstall -y onnxruntime onnxruntime-gpu ; pip install onnxruntime-gpu')

trt, v = ver('tensorrt')
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
