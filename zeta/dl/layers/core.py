# -*- coding: utf-8 -*-

import numpy as np
from .base import Layer
from ..initializers import InitializeWeights as init
from ..optimizers import OptimizationFunction as optimizer
from ..activations import ActivationFunction as activation


class Activation(Layer):

    def __init__(self, function_name, input_shape = None):
        self.input_shape = input_shape
        self.activation_name = function_name
        self.activation_func = activation(self.activation_name)

    def prep_layer(self): pass

    @property
    def output_shape(self):
        return self.input_shape

    def pass_forward(self, input_signal, train_mode = True, **kwargs):
        self.input_signal = input_signal
        return self.activation_func._forward(input_signal)

    def pass_backward(self, grad):
        return grad * self.activation_func._backward(self.input_signal)


class Dense(Layer):

    def __init__(self, units, activation = 'relu', input_shape = None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape

        self.bias = None
        self.weights = None
        self.init_method = None
        self.optimizer_kwargs = None

    def prep_layer(self):
        self.kernel_shape = (self.input_shape[0], self.units)
        self.weights = init(self.weight_initializer).initialize_weights(self.kernel_shape)
        self.bias = np.zeros((1, self.units))

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
        return (self.units,)

    def pass_forward(self, inputs, train_mode = True):
        self.inputs = inputs
        return inputs @ self.weights + self.bias

    def pass_backward(self, grad):
        prev_weights = self.weights

        dweights = self.inputs.T @ grad
        dbias = np.sum(grad, axis = 0, keepdims = True)

        self.weights = optimizer(self.weight_optimizer)._update(self.weights, dweights)
        self.bias = optimizer(self.weight_optimizer)._update(self.bias, dbias)

        return grad @ prev_weights.T


class Dropout(Layer):

    def __init__(self, drop=0.5):
        self.drop = drop
        self.mask = None

    def prep_layer(self): pass

    @property
    def output_shape(self):
        return self.input_shape

    def pass_forward(self, inputs, train_mode = True, **kwargs):
        if 0. < self.drop < 1.:
            keep_prob = (1 - self.drop)
            if train_mode:
                self.mask = np.random.binomial(1, keep_prob, size = inputs.shape) / keep_prob
                keep_prob = self.mask
            return inputs * keep_prob
        else:
            return inputs

    def pass_backward(self, grad):
        if 0. < self.drop < 1.:
            return grad * self.mask
        else:
            return grad


class Flatten(Layer):

    def __init__(self, input_shape = None):
        self.input_shape = input_shape
        self.prev_shape = None

    def prep_layer(self): pass

    @property
    def output_shape(self):
        return (np.prod(self.input_shape),)

    def pass_forward(self, inputs, train_mode = True, **kwargs):
        self.prev_shape = inputs.shape
        return inputs.reshape((inputs.shape[0], -1))

    def pass_backward(self, grad):
        return grad.reshape(self.prev_shape)
