import jax
import jax.numpy as jnp
from resnet18 import ResNet18
import onnx
from onnx import numpy_helper
import numpy as np
import flax


def onnx_list_to_dict(onnx_params):
    onnx_dict = {initializer.name: np.transpose(numpy_helper.to_array(
        initializer)) for initializer in onnx_params.graph.initializer}
    return onnx_dict


def print_onnx_shapes(onnx_params):
    for initializer in onnx_params.graph.initializer:
        array = numpy_helper.to_array(initializer)
        print(f"- Tensor: {initializer.name!r:45} shape={array.shape}")


def print_onnx_dict_shapes(onnx_dict):
    print(list(map(lambda el: (el[0], el[1].shape), onnx_dict.items())))


def copy_batch_norm(onnx_params, flax_params, flax_block, batch_norm_num, onnx_layer_name):
    # print(f'{flax_block}, BatchNorm_{batch_norm_num}')
    # print(flax_params['params'][flax_block][f'BatchNorm_{batch_norm_num}'][
    #     'bias'].shape)
    assert flax_params['params'][flax_block][f'BatchNorm_{batch_norm_num}'][
        'bias'].shape == onnx_params[f'{onnx_layer_name}_beta'].shape
    flax_params['params'][flax_block][f'BatchNorm_{batch_norm_num}']['bias'] = jnp.array(
        onnx_params[f'{onnx_layer_name}_beta'])
    flax_params['params'][flax_block][f'BatchNorm_{batch_norm_num}']['scale'] = jnp.array(
        onnx_params[f'{onnx_layer_name}_gamma'])
    flax_params['batch_stats'][flax_block][f'BatchNorm_{batch_norm_num}']['mean'] = jnp.array(
        onnx_params[f'{onnx_layer_name}_running_mean'])
    flax_params['batch_stats'][flax_block][f'BatchNorm_{batch_norm_num}']['var'] = jnp.array(
        onnx_params[f'{onnx_layer_name}_running_var'])


def copy_stage(onnx_params, flax_params, stage_num):
    block_num = stage_num + (stage_num-2)
    # first block
    # print(flax_params['params'][f'ResnetBlock_{block_num}']['Conv_0'][
    #     'kernel'].shape)
    # print(onnx_params[f'resnetv15_stage{stage_num}_conv0_weight'].shape)
    assert flax_params['params'][f'ResnetBlock_{block_num}']['Conv_0'][
        'kernel'].shape == onnx_params[f'resnetv15_stage{stage_num}_conv0_weight'].shape
    flax_params['params'][f'ResnetBlock_{block_num}']['Conv_0']['kernel'] = jnp.array(
        onnx_params[f'resnetv15_stage{stage_num}_conv0_weight'])
    copy_batch_norm(onnx_params, flax_params, f'ResnetBlock_{block_num}',
                    0, f'resnetv15_stage{stage_num}_batchnorm0')

    assert flax_params['params'][f'ResnetBlock_{block_num}']['Conv_1'][
        'kernel'].shape == onnx_params[f'resnetv15_stage{stage_num}_conv1_weight'].shape
    flax_params['params'][f'ResnetBlock_{block_num}']['Conv_1']['kernel'] = jnp.array(
        onnx_params[f'resnetv15_stage{stage_num}_conv1_weight'])
    copy_batch_norm(onnx_params, flax_params, f'ResnetBlock_{block_num}',
                    1, f'resnetv15_stage{stage_num}_batchnorm1')

    # skip block
    flax_params['params'][f'SkipConnectionBlock_{stage_num-2}']['Conv_0']['kernel'] = jnp.array(
        onnx_params[f'resnetv15_stage{stage_num}_conv2_weight'])
    copy_batch_norm(onnx_params, flax_params, f'SkipConnectionBlock_{stage_num-2}',
                    0, f'resnetv15_stage{stage_num}_batchnorm2')

    # second block
    flax_params['params'][f'ResnetBlock_{block_num+1}']['Conv_0']['kernel'] = jnp.array(
        onnx_params[f'resnetv15_stage{stage_num}_conv3_weight'])
    copy_batch_norm(onnx_params, flax_params, f'ResnetBlock_{block_num+1}',
                    0, f'resnetv15_stage{stage_num}_batchnorm3')
    flax_params['params'][f'ResnetBlock_{block_num+1}']['Conv_1']['kernel'] = jnp.array(
        onnx_params[f'resnetv15_stage{stage_num}_conv4_weight'])
    copy_batch_norm(onnx_params, flax_params, f'ResnetBlock_{block_num+1}',
                    1, f'resnetv15_stage{stage_num}_batchnorm4')


def onnx2flax(onnx_params, flax_params):
    # print_onnx_shapes(onnx_params)
    # print(type(onnx_params.graph.initializer))
    # print_onnx_dict_shapes(onnx_params)
    flax_params = flax.core.frozen_dict.unfreeze(flax_params)

    # initial block
    flax_params['params']['Conv_0']['kernel'] = jnp.array(
        onnx_params['resnetv15_conv0_weight'])

    flax_params['params']['BatchNorm_0']['bias'] = jnp.array(
        onnx_params['resnetv15_batchnorm0_beta'])
    flax_params['params']['BatchNorm_0']['scale'] = jnp.array(
        onnx_params['resnetv15_batchnorm0_gamma'])
    flax_params['batch_stats']['BatchNorm_0']['mean'] = jnp.array(
        onnx_params['resnetv15_batchnorm0_running_mean'])
    flax_params['batch_stats']['BatchNorm_0']['var'] = jnp.array(
        onnx_params['resnetv15_batchnorm0_running_var'])

    # stage 1
    # resnet block 0
    flax_params['params']['ResnetBlock_0']['Conv_0']['kernel'] = jnp.array(
        onnx_params['resnetv15_stage1_conv0_weight'])
    copy_batch_norm(onnx_params, flax_params, 'ResnetBlock_0',
                    0, 'resnetv15_stage1_batchnorm0')
    flax_params['params']['ResnetBlock_0']['Conv_1']['kernel'] = jnp.array(
        onnx_params['resnetv15_stage1_conv1_weight'])
    copy_batch_norm(onnx_params, flax_params, 'ResnetBlock_0',
                    1, 'resnetv15_stage1_batchnorm1')
    # resnet block 1
    flax_params['params']['ResnetBlock_1']['Conv_0']['kernel'] = jnp.array(
        onnx_params['resnetv15_stage1_conv2_weight'])
    copy_batch_norm(onnx_params, flax_params, 'ResnetBlock_1',
                    0, 'resnetv15_stage1_batchnorm2')
    flax_params['params']['ResnetBlock_1']['Conv_1']['kernel'] = jnp.array(
        onnx_params['resnetv15_stage1_conv3_weight'])
    copy_batch_norm(onnx_params, flax_params, 'ResnetBlock_1',
                    1, 'resnetv15_stage1_batchnorm3')

    # stages 2, 3 and 4
    for stage in range(2, 5):
        copy_stage(onnx_params, flax_params, stage)

    # final layers
    flax_params['params']['Dense_0']['kernel'] = jnp.array(
        onnx_params['resnetv15_dense0_weight'])
    flax_params['params']['Dense_0']['bias'] = jnp.array(
        onnx_params['resnetv15_dense0_bias'])

    # assert flax_params['params']['ResnetBlock_0']['Conv_0']['kernel'].shape == onnx_params['resnetv15_conv0_weight'].shape
    # assert (jnp.all(jnp.equal(flax_params['params']['Conv_0']
    #                           ['kernel'], onnx_params['resnetv17_conv0_weight'])))
    return flax.core.frozen_dict.freeze(flax_params)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    model = ResNet18()
    flax_params = model.init(rng, jnp.ones((32, 224, 224, 3)))
    onnx_params = onnx.load("resnet18.onnx")
    onnx_params = onnx_list_to_dict(onnx_params)
    flax_params = onnx2flax(onnx_params, flax_params)
