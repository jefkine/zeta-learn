# -*- coding: utf-8 -*-

import numpy as np

from .base import Layer
from ztlearn.utils import one_hot
from ztlearn.utils import get_sentence_tokens
from ztlearn.initializers import InitializeWeights as init
from ztlearn.optimizers import OptimizationFunction as optimizer


class Embedding(Layer):

    def __init__(self,
                       input_dim,                   # number of unique words in the text dataset
                       output_dim,                  # size of the embedding vectors
                       embeddings_init = 'uniform', # init type for the embedding matrix (weights)
                       input_length    = 10):       # size of input sentences

        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.input_length = input_length
        self.input_shape  = None # required by the base class

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
    def layer_parameters(self):
        return sum([np.prod(param.shape) for param in [self.weights]])

    @property
    def output_shape(self):
        return (self.input_length, self.output_dim)

    def prep_layer(self):
        self.uniques_one_hot = one_hot(np.arange(self.input_dim)) # master one hot matrix
        self.kernel_shape    = (self.input_dim, self.output_dim)
        self.weights         = init(self.weight_initializer).initialize_weights(self.kernel_shape) # embeddings

    # inputs should be gotten from sentences_tokens = get_sentence_tokens(text_input)
    def pass_forward(self, inputs, train_mode = True, **kwargs):
        self.inputs = inputs # tokenized inputs

        embeded_inputs = []
        for _, tokens in enumerate(self.inputs.tolist()):

            for i, word_index in enumerate(tokens):
                embed = np.expand_dims(self.uniques_one_hot[word_index,:], 1).T.dot(self.weights)
                tokens[i] = list(np.array(embed).flat)

            embeded_inputs.append(tokens)

        return np.array(embeded_inputs)

    def pass_backward(self, grad, epoch_num, batch_num, batch_size):
        prev_weights = self.weights

        if self.is_trainable:

            dweights = np.sum(grad @ self.weights.T, axis = 1)
            self.weights = optimizer(self.weight_optimizer).update(self.weights, dweights.T, epoch_num, batch_num, batch_size)

        # endif self.is_trainable

        return grad @ prev_weights.T
