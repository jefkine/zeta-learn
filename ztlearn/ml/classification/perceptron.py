# -*- coding: utf-8 -*-

import numpy as np

from numba import jit, config
from ztlearn.utils import JIT_FLAG, CACHE_FLAG, NOGIL_FLAG

from ztlearn.utils import LogIfBusy
from ztlearn.utils import computebar
from ztlearn.initializers import InitializeWeights as init
from ztlearn.objectives import ObjectiveFunction as objective
from ztlearn.activations import ActivationFunction as activate
from ztlearn.optimizers import OptimizationFunction as optimize
from ztlearn.regularizers import RegularizationFunction as regularize

config.NUMBA_DISABLE_JIT = JIT_FLAG

class Perceptron:

    def __init__(self,
                       epochs,
                       activation     = 'sigmoid',
                       loss           = 'categorical_crossentropy',
                       init_method    = 'he_normal',
                       optimizer      = {},
                       penalty        = 'lasso',
                       penalty_weight = 0,
                       l1_ratio       = 0.5):

        self.epochs         = epochs
        self.activate       = activate(activation)
        self.loss           = objective(loss)
        self.init_method    = init(init_method)
        self.optimizer      = optimizer
        self.regularization = regularize(penalty, penalty_weight, l1_ratio = l1_ratio)

    @LogIfBusy
    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def fit(self, inputs, targets, verbose = False):
        fit_stats = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

        self.weights = self.init_method.initialize_weights((inputs.shape[1], targets.shape[1]))
        self.bias = np.zeros((1, targets.shape[1]))

        for i in range(self.epochs):
            linear_predictions = inputs.dot(self.weights) + self.bias
            predictions        = self.activate.forward(linear_predictions)

            loss = self.loss.forward(predictions, targets) + self.regularization.regulate(self.weights)
            acc  = self.loss.accuracy(predictions, targets)

            fit_stats["train_loss"].append(np.mean(loss))
            fit_stats["train_acc"].append(np.mean(acc))

            grad      = self.loss.backward(predictions, targets) * self.activate.backward(linear_predictions)
            d_weights = inputs.T.dot(grad) + self.regularization.derivative(self.weights)
            d_bias    = np.sum(grad, axis = 0, keepdims = True)

            self.weights = optimize(self.optimizer).update(self.weights, d_weights)
            self.bias    = optimize(self.optimizer).update(self.bias, d_bias)

            if verbose:
                print('TRAINING: Epoch-{} loss: {:2.4f} acc: {:2.4f}'.format(i+1, loss, acc))
            else:
                computebar(self.epochs, i)

        return fit_stats

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def predict(self, inputs):
        # return self.activate.forward(inputs.dot(self.weights) + self.bias)
        return inputs.dot(self.weights) + self.bias

    @property
    def model_weights(self):
        return self.weights
