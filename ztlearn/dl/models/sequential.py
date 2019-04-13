# -*- coding: utf-8 -*-

from .trainer import Trainer
from ztlearn.utils import print_pad
from ztlearn.utils import custom_tuple
from ztlearn.dl.layers import Activation


class Sequential(Trainer):

    __slots__ = ['layers', 'layer_num', 'init_method', 'model_name', 'is_trainable']

    def __init__(self, init_method = 'he_normal', model_name = 'ztlearn_model'):
        self.layers      = []
        self.layer_num   = 0
        self.model_name  = model_name
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
            collection = self.layers[self.layer_num]
        except IndexError:
            raise StopIteration
        self.layer_num += 1

        return collection

    def __str__(self):
        layer_names, layer_params, layer_output = ['LAYER TYPE'], ['PARAMS'], ['OUTPUT SHAPE']

        for _, layer in enumerate(self.layers):
            layer_names.append(layer.layer_name)
            layer_output.append(custom_tuple(layer.output_shape))
            layer_params.append("{:,}".format(layer.layer_parameters))

        max_name   = len(max(layer_names,  key = len))
        max_params = len(max(layer_params, key = len))
        max_output = len(max(layer_output, key = len))

        liner  = [max_name, max_params, max_output]
        lining = ""
        for col_size in liner:
            lining += "+" + ("-" * (col_size + 2))
        lining += "+"

        total_params  = 0
        model_layers  = print_pad(1)+ " " + self.model_name.upper() + print_pad(1)
        model_layers += print_pad(1)+ " Input Shape: " + str(self.layers[0].input_shape) + print_pad(1)
        for i, layer in enumerate(self.layers):
            if i < 2:
                model_layers += lining + print_pad(1)
            model_layers += "¦ {:<{max_name}} ¦ {:>{max_params}} ¦ {:>{max_output}} ¦ ".format(layer_names[i],
                                                                                               layer_params[i],
                                                                                               layer_output[i],
                                                                                               max_name   = max_name,
                                                                                               max_params = max_params,
                                                                                               max_output = max_output)
            model_layers += print_pad(1)

            if i > 0:
                total_params += int(layer_params[i].replace(',', ''))

        model_layers += lining + print_pad(1)
        model_layers += print_pad(1) + " TOTAL PARAMETERS: " + "{:,}".format(total_params) + print_pad(1)

        return model_layers

    def summary(self, model_name = 'ztlearn_model'):
        self.model_name = model_name
        print(self.__str__())

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

    def backward_pass(self, loss_grad, epoch_num, batch_num, batch_size):
        for layer in reversed(self.layers):
            loss_grad = layer.pass_backward(loss_grad, epoch_num, batch_num, batch_size)
