import numpy as np
import torch.nn as nn
import pytest
import tensorflow as tf

from test.utils import convert_and_test
from test.layers.convolutions.conv2d_utils import Conv2dSame


class LayerTest(nn.Module):
    def __init__(self, inp, out, kernel_size=3, padding=1, stride=1, bias=False, dilation=1, groups=1, padding_mode='valid'):
        super(LayerTest, self).__init__()
        assert padding_mode in ['valid', 'same']
        conv_module = nn.Conv2d 
        if padding_mode == 'same':
            conv_module = Conv2dSame
        self.conv = conv_module(
            inp, out, kernel_size=kernel_size, padding=padding,
            stride=stride, bias=bias, dilation=dilation, groups=groups
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def func(change_ordering, kernel_size, padding, stride, bias, dilation, groups, padding_mode):
    if not tf.test.gpu_device_name() and not change_ordering:
        print("skip this 11")
        pytest.skip("Skip! Since tensorflow Conv2D op currently only supports the NHWC tensor format on the CPU")
    if stride > 1 and dilation > 1:
        print("skip this 22")
        pytest.skip("strides > 1 not supported in conjunction with dilation_rate > 1")
    model = LayerTest(
        groups * 3, groups,
        kernel_size=kernel_size, padding=padding,
        stride=stride, bias=bias, dilation=dilation, groups=groups, padding_mode=padding_mode)
    model.eval()
    input_np = np.random.uniform(0, 1, (1, groups * 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering, padding_mode=padding_mode)


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('kernel_size', [1, 3, 5, 7])
@pytest.mark.parametrize('padding', [0, 1, 3, 5])
@pytest.mark.parametrize('stride', [1])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('groups', [1, 2, 3])
@pytest.mark.parametrize('padding_mode', ['valid'])
def test_conv2d_case1(change_ordering, kernel_size, padding, stride, bias, dilation, groups, padding_mode):
    func(change_ordering, kernel_size, padding, stride, bias, dilation, groups, padding_mode)


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('kernel_size', [1, 3, 5, 7])
# @pytest.mark.parametrize('kernel_size', [1, 3])
@pytest.mark.parametrize('padding', [0, 1, 3, 5])
# @pytest.mark.parametrize('padding', [0, 1])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('bias', [True, False])
# @pytest.mark.parametrize('bias', [True])
@pytest.mark.parametrize('dilation', [1])
@pytest.mark.parametrize('groups', [1, 2, 3])
@pytest.mark.parametrize('padding_mode', ['same', 'valid'])
def test_conv2d_case2(change_ordering, kernel_size, padding, stride, bias, dilation, groups, padding_mode):
    func(change_ordering, kernel_size, padding, stride, bias, dilation, groups, padding_mode)
