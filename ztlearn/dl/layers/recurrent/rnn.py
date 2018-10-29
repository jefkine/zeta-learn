# -*- coding: utf-8 -*-

import numpy as np

from ..base import Layer
from ztlearn.utils import clip_gradients as cg
from ztlearn.initializers import InitializeWeights as init
from ztlearn.activations import ActivationFunction as activate
from ztlearn.optimizers import OptimizationFunction as optimizer


class RNN(Layer):

    def __init__(self, h_units, activation = None, bptt_truncate = 5, input_shape = None):
        self.h_units       = h_units # number of hidden states
        self.activation    = activation # should be tanh by default
        self.bptt_truncate = bptt_truncate
        self.input_shape   = input_shape

        self.init_method      = None
        self.optimizer_kwargs = None

        self.W_input  = None
        self.W_output = None
        self.W_recur  = None

        self.b_output = None
        self.b_input  = None

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
    def layer_parameters(self):
        parameters = [
                        self.W_input,
                        self.W_output,
                        self.W_recur,

                        self.b_output,
                        self.b_input
                     ]
        return sum([np.prod(param.shape) for param in parameters])

    @property
    def output_shape(self):
        return self.input_shape

    def prep_layer(self):
        _, input_dim = self.input_shape

        self.W_input  = init(self.init_method).initialize_weights((self.h_units, input_dim))
        self.W_output = init(self.init_method).initialize_weights((input_dim, self.h_units))
        self.W_recur  = init(self.init_method).initialize_weights((self.h_units, self.h_units))

        self.b_output = np.zeros((input_dim,))
        self.b_input  = np.zeros((self.h_units,))

    # implementation based on techniques as seen here: https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/RNNLM.ipynb
    def pass_forward(self, inputs, train_mode = True):
        self.inputs = inputs
        batch_size, time_steps, input_dim = inputs.shape

        self.state_inputs  = np.zeros((batch_size, time_steps, self.h_units))
        self.states        = np.zeros((batch_size, time_steps + 1, self.h_units)) # additional(+1) last column containing the final state also set to zero
        self.state_outputs = np.zeros((batch_size, time_steps, input_dim))

        for t in range(time_steps):
            self.state_inputs[:, t]  = (np.dot(inputs[:, t], self.W_input.T) + np.dot(self.states[:, t - 1], self.W_recur.T)) + self.b_input
            self.states[:, t]        = activate(self.activation).forward(self.state_inputs[:, t])
            self.state_outputs[:, t] = np.dot(self.states[:, t], self.W_output.T) + self.b_output

        if not train_mode:
            return activate('softmax').forward(self.state_outputs) # if mode is not training

        return self.state_outputs

    # implementation based on techniques as seen here: https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/RNNLM.ipynb
    def pass_backward(self, grad):
        _, time_steps, _ = grad.shape
        next_grad        = np.zeros_like(grad)

        if self.is_trainable:

            dW_input  = np.zeros_like(self.W_input)
            dW_recur  = np.zeros_like(self.W_recur)
            dW_output = np.zeros_like(self.W_output)

            db_input  = np.zeros_like(self.b_input)
            db_output = np.zeros_like(self.b_output)

            for t in np.arange(time_steps)[::-1]: # reversed
                dW_output       += np.dot(grad[:, t].T, self.states[:, t])
                db_output       += np.sum(grad[:, t], axis = 0)
                dstate           = np.dot(grad[:, t], self.W_output) * activate(self.activation).backward(self.state_inputs[:, t])
                next_grad[:, t]  = np.dot(dstate, self.W_input)

                for tt in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]: # reversed
                    dW_input += np.dot(dstate.T, self.inputs[:, tt])
                    dW_recur += np.dot(dstate.T, self.states[:, tt - 1])
                    db_input += np.sum(dstate, axis = 0)
                    dstate    = np.dot(dstate, self.W_recur) * activate(self.activation).backward(self.state_inputs[:, tt - 1])

            # optimize weights and bias
            self.W_input  = optimizer(self.optimizer_kwargs).update(self.W_input,  cg(dW_input))
            self.W_output = optimizer(self.optimizer_kwargs).update(self.W_output, cg(dW_output))
            self.W_recur  = optimizer(self.optimizer_kwargs).update(self.W_recur,  cg(dW_recur))

            self.b_input  = optimizer(self.optimizer_kwargs).update(self.b_input,  cg(db_input))
            self.b_output = optimizer(self.optimizer_kwargs).update(self.b_output, cg(db_output))

        # endif self.is_trainable

        return next_grad
