# -*- coding: utf-8 -*-

import numpy as np

from .base import Layer
from ztlearn.utils import one_hot
from ztlearn.initializers import InitializeWeights as init
from ztlearn.optimizers import OptimizationFunction as optimizer

class Embedding(Layer):

    def __init__(self, input_dim, output_dim, activation = 'relu', input_shape = (1,10)):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.input_shape = input_shape

        self.init_method = None
        self.optimizer_kwargs = None

        self.is_trainable = True

    @property
    def trainable(self):
        return self.is_trainable

    @trainable.setter
    def trainable(self, is_trainable):
        self.is_trainable = is_trainable

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
        return self.input_shape

    def prep_layer(self):
        self.kernel_shape = (self.input_dim, self.output_dim)
        self.weights = init(self.weight_initializer).initialize_weights(self.kernel_shape)

    def pass_forward(self, inputs, train_mode = True, **kwargs):
        self.inputs = inputs
        batch_size, num_rows, num_cols = self.inputs.shape
        self.one_hot_inputs = np.zeros((batch_size, num_cols, self.input_dim))

        for i in range(batch_size):
            self.one_hot_inputs[i,:,:] = one_hot(self.inputs[i,:,:], num_classes = self.input_dim)

        return np.expand_dims(np.sum(np.matmul(self.one_hot_inputs, self.weights), axis=2), axis = 1)

    def pass_backward(self, grad):
        d_inputs = np.matmul(grad, self.one_hot_inputs)
        d_embeddings = np.sum(d_inputs, axis = 0)
        self.weights = optimizer(self.weight_optimizer).update(self.weights, d_embeddings.T)
