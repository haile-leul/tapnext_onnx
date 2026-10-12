"""Check everything the export / runtime needs before the long steps.

  python tools/check_env.py [--trackon-root ../track_on]
"""

import argparse
import importlib
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--trackon-root')
  args = ap.parse_args()
  ok = True

  def bad(msg):
    nonlocal ok
    ok = False
    print('  -> ' + msg)

  print('python', sys.version.split()[0], platform.system())
  for mod in ('numpy', 'cv2', 'onnx', 'onnxscript', 'transformers', 'timm', 'yaml'):
    try:
      m = importlib.import_module(mod)
      print(f'{mod:12s} {getattr(m, "__version__", "?")}')
    except Exception as e:  # pylint: disable=broad-except
      print(f'{mod:12s} MISSING ({e})')
      if mod != 'onnxscript':
        bad(f'pip install {"opencv-python" if mod == "cv2" else "pyyaml" if mod == "yaml" else mod}')
  try:
    import transformers  # pylint: disable=g-import-not-at-top
    from transformers import DINOv3ViTModel  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import
  except Exception:  # pylint: disable=broad-except
    bad('transformers >= 4.56 is needed for DINOv3 (pip install transformers==4.56.1)')

  try:
    import torch  # pylint: disable=g-import-not-at-top
    print(f'torch        {torch.__version__} (CUDA build {torch.version.cuda}, available {torch.cuda.is_available()})')
    if torch.cuda.is_available():
      p = torch.cuda.get_device_properties(0)
      free, total = torch.cuda.mem_get_info()
      print(f'  GPU {p.name}, CC {p.major}.{p.minor}, VRAM free {free / 2**30:.1f} / {total / 2**30:.1f} GB')
    else:
      bad('install a CUDA build of torch (see README)')
  except Exception as e:  # pylint: disable=broad-except
    bad(f'torch missing: {e}')

  try:
    import numpy as np  # pylint: disable=g-import-not-at-top
    import onnxruntime as ort  # pylint: disable=g-import-not-at-top
    from onnx import TensorProto, helper  # pylint: disable=g-import-not-at-top
    from trackon_runtime import load_cuda_dlls_for_ort  # pylint: disable=g-import-not-at-top
    load_cuda_dlls_for_ort(ort)
    print(f'onnxruntime  {ort.__version__}')
    g = helper.make_graph([helper.make_node('MatMul', ['a', 'a'], ['b'])], 'g',
                          [helper.make_tensor_value_info('a', TensorProto.FLOAT, [8, 8])],
                          [helper.make_tensor_value_info('b', TensorProto.FLOAT, [8, 8])])
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
    mdl.ir_version = 8
    s = ort.InferenceSession(mdl.SerializeToString(), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    s.run(None, {'a': np.ones((8, 8), np.float32)})
    print('  test session runs on', s.get_providers()[0])
    if s.get_providers()[0] != 'CUDAExecutionProvider':
      bad('onnxruntime CUDA not usable: pip uninstall -y onnxruntime; pip install onnxruntime-gpu '
          '(same CUDA major as torch)')
  except Exception as e:  # pylint: disable=broad-except
    bad(f'onnxruntime problem: {e}')

  try:
    from trackon_runtime import import_tensorrt  # pylint: disable=g-import-not-at-top
    trt = import_tensorrt()
    b = trt.Builder(trt.Logger(trt.Logger.WARNING))
    print(f'tensorrt     {trt.__version__} (builder OK, fast fp16 {b.platform_has_fast_fp16})')
  except Exception as e:  # pylint: disable=broad-except
    bad(f'tensorrt not usable: {e} (set TENSORRT_DIR and install the wheel from <TensorRT>/python)')

  try:
    from huggingface_hub import hf_hub_download  # pylint: disable=g-import-not-at-top
    hf_hub_download('facebook/dinov3-vits16plus-pretrain-lvd1689m', 'config.json')
    print('DINOv3       access OK (facebook/dinov3-vits16plus-pretrain-lvd1689m)')
  except Exception as e:  # pylint: disable=broad-except
    bad('no access to the gated DINOv3 weights: request access on '
        'https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m and run '
        f'`huggingface-cli login` ({type(e).__name__})')

  if args.trackon_root:
    f = os.path.join(args.trackon_root, 'model', 'trackon.py')
    print('track_on repo', 'OK' if os.path.isfile(f) else 'NOT FOUND at ' + args.trackon_root)
    if not os.path.isfile(f):
      ok = False
  print('\nALL GOOD' if ok else '\nfix the items marked -> first')


if __name__ == '__main__':
  main()
