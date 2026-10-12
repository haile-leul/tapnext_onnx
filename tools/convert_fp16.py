"""Weight-only fp16: store large weights as fp16 + a Cast back to fp32.

Halves the file size; all compute, inputs and outputs stay fp32. (For fp16
*compute* on GPU, let TensorRT do it: trtexec --fp16, or ORT's TensorRT EP
with trt_fp16_enable=1, using the fp32 model.)
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1:3]
m = onnx.load(src)
g = m.graph
new_inits, casts = [], []
for init in g.initializer:
  arr = numpy_helper.to_array(init)
  if arr.dtype == np.float32 and arr.size >= 1024:
    name16 = init.name + '__fp16'
    new_inits.append(numpy_helper.from_array(arr.astype(np.float16), name16))
    casts.append(helper.make_node('Cast', [name16], [init.name], to=onnx.TensorProto.FLOAT,
                                  name=init.name + '__cast'))
  else:
    new_inits.append(init)
del g.initializer[:]
g.initializer.extend(new_inits)
nodes = casts + list(g.node)
del g.node[:]
g.node.extend(nodes)
onnx.checker.check_model(m)
onnx.save(m, dst)
print('saved', dst)
