# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class Layer(ABC):

    def __init__(self, layer_name = 'zeta_squential'):
        self.layer_name = layer_name
            
    @property
    def input_shape(self):
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        self.__input_shape = input_shape

    @property
    def output_shape(self):
        return self.input_shape

    @property
    def layer_cls_name(self):
        return self.__class__.__name__

    @abstractmethod
    def pass_forward(self): pass

    @abstractmethod
    def pass_backward(self): pass
