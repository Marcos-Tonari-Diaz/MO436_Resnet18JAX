from functools import partial
from typing import Callable, Tuple
from flax import linen as nn
import jax.numpy as jnp
import jax


class ResnetBlock(nn.Module):
    num_filters: int
    strides: Tuple

    @nn.compact
    def __call__(self, x):
        norm = partial(
            nn.BatchNorm, momentum=0.9, use_running_average=True)
        y = nn.Conv(self.num_filters, (1, 1))(x)
        y = norm()(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_filters, (3, 3), self.strides)(y)
        y = norm()(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_filters * 4, (1, 1))(y)
        y = norm()(y)
        return y


class SkipConnectionBlock(nn.Module):
    num_filters: int
    strides: Tuple

    @nn.compact
    def __call__(self, x):
        norm: Callable = partial(
            nn.BatchNorm, momentum=0.9, use_running_average=True)
        y = nn.Conv(self.num_filters, (1, 1))(x)
        y = norm()(y)
        return y


class ResNet50(nn.Module):
    @nn.compact
    def __call__(self, x):
        y = nn.Conv(64, (7, 7), (2, 2), padding=[
                    (3, 3), (3, 3)], use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=True, momentum=0.9)(y)
        y = nn.relu(y)
        y = nn.max_pool(y, (3, 3), strides=(2, 2), padding='SAME')

        y = ResnetBlock(64, strides=(1, 1))(y)
        + SkipConnectionBlock(256, strides=(1, 1))(y)
        y = nn.relu(y)
        for _ in range(2):
            y = ResnetBlock(64, strides=(1, 1))(y) + y
            y = nn.relu(y)

        y = ResnetBlock(64, strides=(2, 2))(y)
        + SkipConnectionBlock(256, strides=(2, 2))(y)
        y = nn.relu(y)
        for _ in range(3):
            y = ResnetBlock(64, strides=(1, 1))(y) + y
            y = nn.relu(y)

        y = ResnetBlock(64, strides=(2, 2))(y)
        + SkipConnectionBlock(256, strides=(2, 2))(y)
        y = nn.relu(y)
        for _ in range(5):
            y = ResnetBlock(64, strides=(1, 1))(y) + y
            y = nn.relu(y)

        y = ResnetBlock(64, strides=(2, 2))(y)
        + SkipConnectionBlock(256, strides=(2, 2))(y)
        y = nn.relu(y)
        for _ in range(2):
            y = ResnetBlock(64, strides=(1, 1))(y) + y
            y = nn.relu(y)

        y = jnp.mean(y, axis=(1, 2))
        y = nn.Dense(1000)(y)
        y = jnp.asarray(y)
        return y


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    model = ResNet50()
    params = model.init(rng, jnp.ones((32, 224, 224, 3)))
    params_shapes = jax.tree_util.tree_map(lambda x: x.shape, params['params'])
    print(params_shapes)
    # print(params_shapes['Conv_0'].keys())
    # print("batch norm")
    # batchparams_shapes = jax.tree_util.tree_map(
    #     lambda x: x.shape, params['batch_stats'])
    # print(batchparams_shapes)
