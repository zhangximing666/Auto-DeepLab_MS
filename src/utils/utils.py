"""Utilities"""
import random
import numpy as np
import cv2

import mindspore
import mindspore.nn as nn
import mindspore.numpy as msnp
import mindspore.ops as ops
from numpy.lib.function_base import flip


def fast_hist(predict, label, n):
    """
    fast_hist
    inputs:
        - predict (ndarray)
        - label (ndarray)
        - n (int) - number of classes
    outputs:
        - fast histogram
    """
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(np.int32) + predict[k], minlength=n ** 2).reshape(n, n)


def rescale_batch(inputs, new_scale):
    """
        inputs:
            - inputs (ndarray, shape=(n, c, h, w))
            - new_scale
        outputs: ndarray, shape=(n, c, new_scale[0], new_scale[1])
    """
    n, c, _, _ = inputs.shape
    # n, c, h, w -> n, h, w, c
    input_batch = inputs.transpose((0, 2, 3, 1))
    scaled_batch = np.zeros((n, new_scale[0], new_scale[1], c))
    for i in range(n):
        scaled_batch[i] = cv2.resize(input_batch[i], (new_scale[1], new_scale[0]), interpolation=cv2.INTER_CUBIC)
    scaled_batch = np.ascontiguousarray(scaled_batch)
    # n, h, w, c -> n, c, h, w
    scaled_batch = scaled_batch.transpose((0, 3, 1, 2))
    return scaled_batch


class BuildEvalNetwork(nn.Cell):
    """BuildEvalNetwork"""
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        """construct"""
        output = self.network(input_data)
        output = self.softmax(output)
        return output


class InferWithFlipNetwork(nn.Cell):
    """InferWithFlipNetwork"""
    def __init__(self, network, flip=True, input_format="NCHW"):
        super(InferWithFlipNetwork, self).__init__()
        self.eval_net = BuildEvalNetwork(network)
        self.transpose = ops.Transpose()
        self.flip = flip
        self.format = input_format

    def construct(self, input_data):
        """construct"""
        if self.format == "NHWC":
            input_data = self.transpose(input_data, (0, 3, 1, 2))
        output = self.eval_net(input_data)

        if self.flip:
            flip_input = msnp.flip(input_data, 3)
            flip_output = self.eval_net(flip_input)
            output += msnp.flip(flip_output, 3)
        
        return output


def prepare_seed(seed):
    """prepare_seed"""
    mindspore.set_seed(seed)
    random.seed(seed)
