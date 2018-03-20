# -*- coding: utf-8 -*-

import numpy as np
from .base import Layer
from ztlearn.utils import get_pad
from ztlearn.utils import im2col_indices
from ztlearn.utils import col2im_indices
from ..initializers import InitializeWeights as init
from ..optimizers import OptimizationFunction as optimizer

class Conv2D(Layer):

    def __init__(self, filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (1, 8, 8), strides = (1, 1), padding = 'valid'):
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_size = kernel_size
        self.input_shape = input_shape

        self.init_method = None
        self.optimizer_kwargs = None

        self.stride_height, self.stride_width = self.strides
        self.kernel_height, self.kernel_width = self.kernel_size
        self.input_channels, self.input_height, self.input_width = self.input_shape

    def prep_layer(self):
        self.kernel_shape = (self.filters, self.input_shape[0], self.kernel_size[0], self.kernel_size[1])
        self.weights = init(self.weight_initializer).initialize_weights(self.kernel_shape)
        self.bias = np.zeros((self.kernel_shape[0], 1))

    @property
    def weight_initializer(self):
        return self.init_method

    @weight_initializer.setter
    def weight_initializer(self, init_method):
        self.init_method = init_method

    @property
    def weight_optimizer(self):
        return self.optimizer_kwargs

    @weight_optimizer.setter
    def weight_optimizer(self, optimizer_kwargs = {}):
        self.optimizer_kwargs = optimizer_kwargs

    @property
    def layer_activation(self):
        return self.activation

    @layer_activation.setter
    def layer_activation(self, activation):
        self.activation = activation

    @property
    def output_shape(self):
        pad_height, pad_width = get_pad(self.padding, self.input_height, self.input_width, self.stride_height, self.stride_width, self.kernel_height, self.kernel_width)
        output_height = (self.input_height + np.sum(pad_height) - self.kernel_height) / self.stride_height + 1
        output_width = (self.input_width + np.sum(pad_width) - self.kernel_width) / self.stride_width + 1
        return self.filters, int(output_height), int(output_width)

    def pass_forward(self, inputs, train_mode = True, **kwargs):
        self.filter_num, self.filter_depth, self.filter_height, self.filter_width = self.weights.shape
        input_num, input_depth, input_height, input_width = inputs.shape
        self.input_shape = inputs.shape
        self.inputs = inputs

        pad_height, pad_width = get_pad(self.padding, input_height, input_width, self.stride_height, self.stride_width, self.kernel_height, self.kernel_width)

        # confirm dimensions
        assert (input_height + np.sum(pad_height) - self.filter_height) % self.stride_height == 0, 'height does not work'
        assert (input_width + np.sum(pad_width) - self.filter_width) %  self.stride_width == 0, 'width does not work'

        # generate output
        output_height = (input_height + np.sum(pad_height) - self.filter_height) / self.stride_height + 1
        output_width = (input_width + np.sum(pad_width) - self.filter_width) / self.stride_width + 1

        # convert to columns
        self.input_col = im2col_indices(inputs, self.filter_height, self.filter_width, padding = (pad_height, pad_width), stride = 1)
        self.weight_col =  self.weights.reshape(self.filter_num, -1)

        # calculate ouput
        output = self.weight_col @ self.input_col + self.bias
        output = output.reshape(self.filter_num, int(output_height), int(output_width), input_num)

        return output.transpose(3, 0, 1, 2)

    def pass_backward(self, grad):
        input_num, input_depth, input_height, input_width = self.input_shape

        dbias = np.sum(grad, axis = (0, 2, 3))
        dbias = dbias.reshape(self.filter_num, -1)

        doutput_reshaped = grad.transpose(1, 2, 3, 0).reshape(self.filter_num, -1)

        dweights = doutput_reshaped @ self.input_col.T
        dweights = dweights.reshape(self.weights.shape)

        # optimize the weights and bias
        self.weights = optimizer(self.weight_optimizer)._update(self.weights, dweights)
        self.bias = optimizer(self.weight_optimizer)._update(self.bias, dbias)

        weight_reshape = self.weights.reshape(self.filter_num, -1)
        dinput_col = weight_reshape.T @ doutput_reshaped

        pad_height, pad_width = get_pad(self.padding, input_height, input_width, self.stride_height, self.stride_width, self.kernel_height, self.kernel_width)
        dinputs = col2im_indices(dinput_col, self.input_shape, self.filter_height, self.filter_width, padding = (pad_height, pad_width), stride = self.stride_height)

        return dinputs
