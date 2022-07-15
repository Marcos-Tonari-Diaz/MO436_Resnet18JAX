
import onnx
from onnx import numpy_helper

model = onnx.load("resnet18-v1-7.onnx")

for initializer in model.graph.initializer:
    array = numpy_helper.to_array(initializer)
    print(f"- Tensor: {initializer.name!r:45} shape={array.shape}")
    # print(array)
