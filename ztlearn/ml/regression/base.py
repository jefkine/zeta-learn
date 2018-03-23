# -*- coding: utf-8 -*-

import numpy as np

from ztlearn.utils import LogIfBusy
from ztlearn.dl.initializers import InitializeWeights as init
from ztlearn.dl.objectives import ObjectiveFunction as objective
from ztlearn.dl.optimizers import OptimizationFunction as optimize
from ..regularizers import RegularizationFunction as regularize


class Regression(object):

    def __init__(self, epochs,
                       loss = 'mean_squared_error',
                       init_method = 'he_uniform',
                       optimizer = {},
                       penalty = 'ridge',
                       penalty_weight = 0.5,
                       l1_ratio = 0.5):
        self.epochs = epochs
        self.loss = objective(loss)
        self.init_method = init(init_method)
        self.optimizer = optimize(optimizer)
        self.regularization = regularize(penalty, penalty_weight, l1_ratio = l1_ratio)

    @LogIfBusy
    def fit(self, inputs, targets, verbose = True):
        fit_stats = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
        inputs = np.column_stack((np.ones(inputs.shape[0]), inputs))
        self.weights = self.init_method.initialize_weights((inputs.shape[1], ))

        for i in range(self.epochs):
            predictions = inputs.dot(self.weights)
            mse = self.loss.forward(np.expand_dims(predictions, axis = 1), np.expand_dims(targets, axis = 1)) + self.regularization.regulate(self.weights)
            acc = self.loss.accuracy(predictions, targets)

            fit_stats["train_loss"].append(np.mean(mse))
            fit_stats["train_acc"].append(np.mean(acc))

            if verbose:
                print('TRAINING: Epoch-{} loss: {:.2f} acc: {:.2f}'.format(i+1, mse, acc))

            cost_gradient = self.loss.backward(predictions, targets)
            d_weights = cost_gradient.dot(inputs) + self.regularization.derivative(self.weights)
            self.weights = self.optimizer.update(self.weights, d_weights)

        return fit_stats

    def predict(self, inputs):
        inputs = np.column_stack((np.ones(inputs.shape[0]), inputs))
        return inputs.dot(self.weights)
