# -*- coding: utf-8 -*-

import numpy as np
from numba import jit

from ztlearn.utils import LogIfBusy
from ztlearn.utils import computebar
from ztlearn.initializers import InitializeWeights as init
from ztlearn.objectives import ObjectiveFunction as objective
from ztlearn.optimizers import OptimizationFunction as optimize
from ztlearn.activations import ActivationFunction as activation
from ztlearn.regularizers import RegularizationFunction as regularize


class LogisticRegression:

    def __init__(self,
                       epochs,
                       loss           = 'binary_crossentropy',
                       init_method    = 'he_normal',
                       optimizer      = {},
                       penalty        = 'lasso',
                       penalty_weight = 0,
                       l1_ratio       = 0.5):

        self.epochs         = epochs
        self.loss           = objective(loss)
        self.init_method    = init(init_method)
        self.optimizer      = optimize(optimizer)
        self.activate       = activation('sigmoid')
        self.regularization = regularize(penalty, penalty_weight, l1_ratio = l1_ratio)

    @LogIfBusy
    @jit(nogil = True, cache = True)
    def fit(self, inputs, targets, verbose = False):
        fit_stats    = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
        self.weights = self.init_method.initialize_weights((inputs.shape[1], ))

        for i in range(self.epochs):
            predictions = self.activate.forward(inputs.dot(self.weights))
            cost        = self.loss.forward(np.expand_dims(predictions, axis = 1), np.expand_dims(targets, axis = 1)) + self.regularization.regulate(self.weights)
            acc         = self.loss.accuracy(predictions, targets)

            fit_stats["train_loss"].append(np.mean(cost))
            fit_stats["train_acc"].append(np.mean(acc))

            cost_gradient = self.loss.backward(predictions, targets)
            d_weights     = inputs.T.dot(cost_gradient) + self.regularization.derivative(self.weights)
            self.weights  = self.optimizer.update(self.weights, d_weights)

            if verbose:
                print('TRAINING: Epoch-{} loss: {:.2f} acc: {:.2f}'.format(i+1, cost, acc))
            else:
                computebar(self.epochs, i)

        return fit_stats

    @LogIfBusy
    @jit(nogil = True, cache = True)
    def fit_NR(self, inputs, targets, verbose = False):
        ''' Newton-Raphson Method '''
        fit_stats = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
        self.weights = self.init_method.initialize_weights((inputs.shape[1], ))

        for i in range(self.epochs):
            predictions = self.activate.forward(inputs.dot(self.weights))
            cost        = self.loss.forward(np.expand_dims(predictions, axis = 1), np.expand_dims(targets, axis = 1)) + self.regularization.regulate(self.weights)
            acc         = self.loss.accuracy(predictions, targets)

            fit_stats["train_loss"].append(np.mean(cost))
            fit_stats["train_acc"].append(np.mean(acc))

            diag_grad     = np.diag(self.activate.backward(inputs.dot(self.weights)))
            # self.weights += np.linalg.pinv(inputs.T.dot(diag_grad).dot(inputs) + self.regularization.derivative(self.weights)).dot(inputs.T).dot((targets - predictions))
            self.weights += np.linalg.pinv(inputs.T.dot(diag_grad).dot(inputs) + self.regularization.derivative(self.weights)).dot(inputs.T.dot(diag_grad)).dot((targets - predictions))

            if verbose:
                print('TRAINING: Epoch-{} loss: {:2.4f} acc: {:2.4f}'.format(i+1, cost, acc))
            else:
                computebar(self.epochs, i)

        return fit_stats

    @jit(nogil = True, cache = True)
    def predict(self, inputs):
        return np.round(self.activate.forward(inputs.dot(self.weights))).astype(int)
