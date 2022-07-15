import jax
import jax.numpy as jnp
from matplotlib.pyplot import axis
from resnet18 import ResNet18
import onnx
from onnx2flax import onnx2flax, onnx_list_to_dict
from image_loader import load_imagenet_val
import functools
import time


def perf_timer_jax(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs).block_until_ready()
        print(f"({func.__name__}) value: {value}")
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"({func.__name__}) Elapsed time: {elapsed_time:2.10f} seconds")
        return value
    return wrapper_timer


@jax.vmap
def test_batch(batch):
    return model.apply(flax_params, batch).block_until_ready()


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
    tic = time.perf_counter()
    for batch in (val_ds):
        res = model.apply(flax_params, jnp.array(batch)).block_until_ready()
        res_list.append(res)
    toc = time.perf_counter()
    elapsed_time = toc - tic
    print(f"JAX Elapsed time: {elapsed_time:2.10f} seconds")

    # jit version
    val_ds = load_imagenet_val("work/datasets/imagenet/val")
    jit_apply = jax.jit(model.apply)
    jit_apply(flax_params, dummy)

    res_list = []
    tic = time.perf_counter()
    for batch in (val_ds):
        res = jit_apply(flax_params, jnp.array(batch)).block_until_ready()
        res_list.append(res)
    toc = time.perf_counter()
    elapsed_time = toc - tic
    print(f"JIT Elapsed time: {elapsed_time:2.10f} seconds")

    # vmap version
    # val_ds = load_imagenet_val("work/datasets/imagenet/val")
    # batch_list = []
    # for batch in (val_ds):
    #     batch_list.append(batch)
    # batch_list = jnp.array(batch_list)
    # tic = time.perf_counter()
    # res_list = test_batch(batch_list)
    # toc = time.perf_counter()
    # elapsed_time = toc - tic
    # print(f"VMAP Elapsed time: {elapsed_time:2.10f} seconds")
