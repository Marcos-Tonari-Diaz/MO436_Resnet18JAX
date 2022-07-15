import jax
import jax.numpy as jnp
from matplotlib.pyplot import axis
from resnet18 import ResNet18
import onnx
from onnx2flax import onnx2flax, onnx_list_to_dict
from image_loader import load_imagenet_val
import numpy as np


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    model = ResNet18()
    dummy = jnp.ones((32, 224, 224, 3))
    flax_params = model.init(rng, dummy)
    onnx_params = onnx.load("resnet18.onnx")
    onnx_params = onnx_list_to_dict(onnx_params)
    flax_params = onnx2flax(onnx_params, flax_params)

    val_ds = load_imagenet_val("work/datasets/imagenet/val")
    res_list = []
    for i, batch in enumerate(val_ds):
        res = model.apply(flax_params, batch)
        res_list.append(res)
    res_arr = jnp.array(res_list)
    res_arr = res_arr.reshape((-1, res_arr.shape[-1]))
    print(res_arr.shape)

    res_arr = np.array(jax.device_get(res_arr))+1
    res_arr = np.argmax(res_arr, axis=1)
    print(res_arr.shape)

    with open("work/scripts/ground_truth_imagenet_padded.txt") as file:
        ground_truth = file.readlines()
        ground_truth = [int(val.rstrip()) for val in ground_truth]
    ground_truth = np.array(ground_truth)
    print(ground_truth.shape)
    print(res_arr[:20])
    print(np.where(res_arr == 171))
    # print(ground_truth[:20])

    print(f'acc: {np.sum(res_arr == ground_truth)/len(res_arr)}')
