# -*- coding: utf-8 -*-

import numpy as np

from .base import Layer
from ztlearn.utils import get_pad
from ztlearn.utils import unroll_inputs
from ztlearn.utils import im2col_indices
from ztlearn.utils import col2im_indices
from ztlearn.utils import get_output_dims
from ztlearn.initializers import InitializeWeights as init
from ztlearn.optimizers import OptimizationFunction as optimizer


class Conv(Layer):

    def __init__(self,
                       filters     = 32,
                       kernel_size = (3, 3),
                       activation  = None,
                       input_shape = (1, 8, 8),
                       strides     = (1, 1),
                       padding     = 'valid'):

        self.filters     = filters
        self.strides     = strides
        self.padding     = padding
        self.activation  = activation
        self.kernel_size = kernel_size
        self.input_shape = input_shape

        self.init_method      = None
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
        pad_height, pad_width = get_pad(self.padding,
                                                      self.input_shape[1],
                                                      self.input_shape[2],
                                                      self.strides[0],
                                                      self.strides[1],
                                                      self.kernel_size[0],
                                                      self.kernel_size[1])

        output_height, output_width = get_output_dims(self.input_shape[1],
                                                                           self.input_shape[2],
                                                                           self.kernel_size,
                                                                           self.strides,
                                                                           self.padding)

        return self.filters, int(output_height), int(output_width)

    def prep_layer(self):
        self.kernel_shape = (self.filters, self.input_shape[0], self.kernel_size[0], self.kernel_size[1])
        self.weights      = init(self.weight_initializer).initialize_weights(self.kernel_shape)
        self.bias         = np.zeros((self.kernel_shape[0], 1))


class Conv2D(Conv):

    def __init__(self,
                       filters     = 32,
                       kernel_size = (3, 3),
                       activation  = None,
                       input_shape = (1, 8, 8),
                       strides     = (1, 1),
                       padding     = 'valid'):

        super(Conv2D, self).__init__(filters, kernel_size, activation, input_shape, strides, padding)

    def pass_forward(self, inputs, train_mode = True, **kwargs):
        self.filter_num, _, _, _  = self.weights.shape
        self.input_shape          = inputs.shape
        self.inputs               = inputs

        input_num, input_depth, input_height, input_width = inputs.shape

        pad_height, pad_width = get_pad(self.padding,
                                                      input_height,
                                                      input_width,
                                                      self.strides[0],
                                                      self.strides[1],
                                                      self.kernel_size[0],
                                                      self.kernel_size[1])

        # confirm dimensions
        assert (input_height + np.sum(pad_height) - self.kernel_size[0]) % self.strides[0] == 0, 'height does not work'
        assert (input_width + np.sum(pad_width) - self.kernel_size[1]) %  self.strides[1]  == 0, 'width does not work'

        # compute output_height and output_width
        output_height, output_width = get_output_dims(input_height, input_width, self.kernel_size, self.strides, self.padding)

        # convert to columns
        self.input_col = im2col_indices(inputs,
                                                self.kernel_size[0],
                                                self.kernel_size[1],
                                                padding = (pad_height, pad_width),
                                                stride  = 1)

        self.weight_col =  self.weights.reshape(self.filter_num, -1)

        # calculate ouput
        output = self.weight_col @ self.input_col + self.bias
        output = output.reshape(self.filter_num, int(output_height), int(output_width), input_num)

        return output.transpose(3, 0, 1, 2)

    def pass_backward(self, grad):
        input_num, input_depth, input_height, input_width = self.input_shape
        doutput_reshaped = grad.transpose(1, 2, 3, 0).reshape(self.filter_num, -1)

        if self.is_trainable:

            dbias = np.sum(grad, axis = (0, 2, 3))
            dbias = dbias.reshape(self.filter_num, -1)

            dweights = doutput_reshaped @ self.input_col.T
            dweights = dweights.reshape(self.weights.shape)

            # optimize the weights and bias
            self.weights = optimizer(self.weight_optimizer).update(self.weights, dweights)
            self.bias    = optimizer(self.weight_optimizer).update(self.bias, dbias)

        # endif self.is_trainable

        weight_reshape = self.weights.reshape(self.filter_num, -1)
        dinput_col     = weight_reshape.T @ doutput_reshaped

        pad_height, pad_width = get_pad(self.padding,
                                                      input_height,
                                                      input_width,
                                                      self.strides[0],
                                                      self.strides[1],
                                                      self.kernel_size[0],
                                                      self.kernel_size[1])

        dinputs = col2im_indices(dinput_col,
                                             self.input_shape,
                                             self.kernel_size[0],
                                             self.kernel_size[1],
                                             padding = (pad_height, pad_width),
                                             stride = self.strides[0])

        return dinputs


class ConvLoop2D(Conv):

    def __init__(self,
                       filters     = 32,
                       kernel_size = (3, 3),
                       activation  = None,
                       input_shape = (1, 8, 8),
                       strides     = (1, 1),
                       padding     = 'valid'):

        super(ConvLoop2D, self).__init__(filters, kernel_size, activation, input_shape, strides, padding)

    def pass_forward(self, inputs, train_mode = True, **kwargs):
        self.filter_num, _, _, _ = self.weights.shape
        self.input_shape         = inputs.shape
        self.inputs              = inputs

        input_num, input_depth, input_height, input_width = inputs.shape

        pad_height, pad_width = get_pad(self.padding,
                                                      input_height,
                                                      input_width,
                                                      self.strides[0],
                                                      self.strides[1],
                                                      self.kernel_size[0],
                                                      self.kernel_size[1])

        x_padded = np.pad(self.inputs, ((0, 0), (0, 0), pad_height, pad_width), mode = 'constant')

        # confirm dimensions
        assert (input_height + np.sum(pad_height) - self.kernel_size[0]) % self.strides[0] == 0, 'height does not work'
        assert (input_width + np.sum(pad_width) - self.kernel_size[1]) %  self.strides[1]  == 0, 'width does not work'

        # compute output_height and output_width
        output_height, output_width = get_output_dims(input_height, input_width, self.kernel_size, self.strides, self.padding)

        output = np.zeros((input_num, self.filter_num, output_height, output_width))

        # convolutions
        for b in np.arange(input_num): # batch number
            for f in np.arange(self.filter_num): # filter number
                for h in np.arange(output_height): # output height
                    for w in np.arange(output_width): # output width
                        h_stride, w_stride = h * self.strides[0], w * self.strides[1]
                        x_patch            = x_padded[b, :, h_stride: h_stride + self.kernel_size[0],
                                                            w_stride: w_stride + self.kernel_size[1]]
                        output[b, f, h, w] = np.sum(x_patch * self.weights[f]) + self.bias[f]

        return output

    def pass_backward(self, grad):
        input_num, input_depth, input_height, input_width = self.inputs.shape

        # initialize the gradient(s)
        dinputs = np.zeros(self.inputs.shape)

        if self.is_trainable:

            # initialize the gradient(s)
            dweights = np.zeros(self.weights.shape)
            dbias    = np.zeros(self.bias.shape)

            pad_height, pad_width = get_pad(self.padding,
                                                          input_height,
                                                          input_width,
                                                          self.strides[0],
                                                          self.strides[1],
                                                          self.kernel_size[0],
                                                          self.kernel_size[1])

            pad_size = (np.sum(pad_height)/2).astype(int)
            if pad_size != 0:
                grad = grad[:, :, pad_size: -pad_size, pad_size: -pad_size]

            # dweights
            for f in np.arange(self.filter_num): # filter number
                for c in np.arange(input_depth): # input depth (channels)
                    for h in np.arange(self.kernel_size[0]): # kernel height
                        for w in np.arange(self.kernel_size[1]): # kernel width
                            input_patch = self.inputs[:,
                                                      c,
                                                      h: input_height - self.kernel_size[0] + h + 1: self.strides[0],
                                                      w: input_width - self.kernel_size[1] + w + 1: self.strides[1]]

                            grad_patch           = grad[:, f]
                            dweights[f, c, h, w] = np.sum(input_patch * grad_patch) / input_num

            # dbias
            for f in np.arange(self.filter_num): # filter number
                dbias[f] = np.sum(grad[:, f]) / input_num

            # optimize the weights and bias
            self.weights = optimizer(self.weight_optimizer).update(self.weights, dweights)
            self.bias    = optimizer(self.weight_optimizer).update(self.bias, dbias)

        # endif self.is_trainable

        # dinputs
        for b in np.arange(input_num): # batch number
            for f in np.arange(self.filter_num): # filter number
                for c in np.arange(input_depth): # input depth (channels)
                    for h in np.arange(self.kernel_size[0]): # kernel height
                        for w in np.arange(self.kernel_size[1]): # kernel width
                            h_stride, w_stride = h * self.strides[0], w * self.strides[1]
                            dinputs[b,
                                    c,
                                    h_stride: h_stride + self.kernel_size[0],
                                    w_stride: w_stride + self.kernel_size[1]] += self.weights[f, c] * grad[b, f, h, w]

        return dinputs


class ConvToeplitzMat(Conv):

    def __init__(self,
                       filters     = 32,
                       kernel_size = (3, 3),
                       activation  = None,
                       input_shape = (1, 8, 8),
                       strides     = (1, 1),
                       padding     = 'valid'):

        super(ConvToeplitzMat, self).__init__(filters, kernel_size, activation, input_shape, strides, padding)

    def pass_forward(self, inputs, train_mode = True, **kwargs):
        self.filter_num, _, _, _ = self.weights.shape
        self.input_shape         = inputs.shape
        self.inputs              = inputs

        input_num, input_depth, input_height, input_width = inputs.shape

        pad_height, pad_width = get_pad(self.padding,
                                                      input_height,
                                                      input_width,
                                                      self.strides[0],
                                                      self.strides[1],
                                                      self.kernel_size[0],
                                                      self.kernel_size[1])

        x_padded = np.pad(self.inputs, ((0, 0), (0, 0), pad_height, pad_width), mode = 'constant')

        # confirm dimensions
        assert (input_height + np.sum(pad_height) - self.kernel_size[0]) % self.strides[0] == 0, 'height does not work'
        assert (input_width + np.sum(pad_width) - self.kernel_size[1]) %  self.strides[1]  == 0, 'width does not work'

        # compute output_height and output_width
        output_height, output_width = get_output_dims(input_height, input_width, self.kernel_size, self.strides, self.padding)

        output = np.zeros((input_num, self.filter_num, output_height, output_width))

        self.input_col = unroll_inputs(x_padded,
                                                 x_padded.shape[0],
                                                 x_padded.shape[1],
                                                 output_height,
                                                 output_width,
                                                 self.kernel_size[0])

        #TODO: weights need to be rearraged in a way to have a matrix
        #      multiplication with the generated toeplitz matrix
        self.weight_col = self.weights.reshape(self.filter_num, -1)

        # calculate ouput
        output = self.weight_col @ self.input_col + self.bias
        # output = np.matmul(self.weight_col, self.input_col) + self.bias
        output = output.reshape(self.filter_num, int(output_height), int(output_width), input_num)

        return output.transpose(3, 0, 1, 2)

    def pass_backward(self, grad): pass
