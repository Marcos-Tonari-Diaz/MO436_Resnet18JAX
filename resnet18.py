from typing import Tuple
from flax import linen as nn
import jax.numpy as jnp
import jax


class ResnetBlock(nn.Module):
    num_filters: int
    strides: Tuple

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.num_filters, (3, 3), self.strides, use_bias=False)(x)
        y = nn.BatchNorm(momentum=0.9, use_running_average=True)(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_filters, (3, 3), use_bias=False)(y)
        y = nn.BatchNorm(
            momentum=0.9, use_running_average=True)(y)
        return y


class SkipConnectionBlock(nn.Module):
    num_filters: int
    strides: Tuple

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.num_filters, (1, 1),
                    strides=(2, 2), use_bias=False)(x)
        y = nn.BatchNorm(momentum=0.9, use_running_average=True)(y)
        return y


def resnet_stage(num_filters, y):
    y_copy = y
    y = ResnetBlock(num_filters, strides=(2, 2))(y)
    y += SkipConnectionBlock(num_filters, strides=(2, 2))(y_copy)
    y = nn.relu(y)
    y = ResnetBlock(num_filters, strides=(1, 1))(y) + y
    y = nn.relu(y)
    return y


class ResNet18(nn.Module):
    @nn.compact
    def __call__(self, x):
        # initial blocks
        y = nn.Conv(64, (7, 7), (2, 2), padding=[
                    (3, 3), (3, 3)], use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=True, momentum=0.9)(y)
        y = nn.relu(y)
        y = nn.max_pool(y, (3, 3), strides=(2, 2), padding='SAME')

        # stage 1
        y = ResnetBlock(64, strides=(1, 1))(y) + y
        y = nn.relu(y)
        y = ResnetBlock(64, strides=(1, 1))(y) + y
        y = nn.relu(y)

        num_filters = 128
        for _ in range(3):
            y = resnet_stage(num_filters, y)
            num_filters *= 2

        y = jnp.mean(y, axis=(1, 2))
        # print(y.shape)
        y = nn.Dense(1000)(y)
        y = nn.softmax(y)
        y = jnp.asarray(y)
        return y


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    model = ResNet18()
    params = model.init(rng, jnp.ones((32, 224, 224, 3)))
    params_shapes = jax.tree_util.tree_map(lambda x: x.shape, params['params'])
    print(params_shapes)
    # print(params_shapes['Conv_0'].keys())
    # print("batch norm")
    # batchparams_shapes = jax.tree_util.tree_map(
    #     lambda x: x.shape, params['batch_stats'])
    # print(batchparams_shapes)
