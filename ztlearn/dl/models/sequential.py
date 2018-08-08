# -*- coding: utf-8 -*-

import numpy as np

from .trainer import Trainer
from ztlearn.dl.layers import Activation


class Sequential(Trainer):

    __slots__ = ['layers', 'layer_num', 'init_method', 'is_trainable']
    def __init__(self, init_method = 'he_normal'):
        self.layers      = []
        self.layer_num   = 0
        self.init_method = init_method

        self.is_trainable = True

    @property
    def trainable(self):
        return self.is_trainable

    @trainable.setter
    def trainable(self, is_trainable):
        self.is_trainable = is_trainable
        for layer in self.layers:
            layer.trainable = self.is_trainable

    @property
    def added_layers(self):
        return self.layers

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.layers[self.layer_num]
        except IndexError:
            raise StopIteration
        self.layer_num += 1

        return result

    def __str__(self):
        model_layers = ""
        for i, layer in enumerate(self.layers):
            model_layers += "LAYER {}: {}\n".format(i + 1, layer.layer_cls_name)

        return model_layers

    def add(self, layer):
        if self.layers:
            layer.input_shape = self.layers[-1].output_shape

        if hasattr(layer, 'weight_initializer'):
            layer.weight_initializer = self.init_method
        self.append_layer(layer)

        if hasattr(layer, 'layer_activation') and layer.layer_activation is not None:
            self.append_layer(Activation(layer.layer_activation, input_shape = self.layers[-1].output_shape))

    def append_layer(self, layer):
        layer.prep_layer()
        self.layers.append(layer)

    def compile(self, loss = 'categorical_crossentropy', optimizer = {}):
        self.loss = loss
        for layer in self.layers:
            if hasattr(layer, 'weight_optimizer'):
                layer.weight_optimizer = optimizer

    def foward_pass(self, inputs, train_mode = False):
        layer_output = inputs
        for layer in self.layers:
            layer_output = layer.pass_forward(layer_output, train_mode)
        return layer_output

    def backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.pass_backward(loss_grad)
