# -*- coding: utf-8 -*-

import numpy as np

from zeta.utils import LogIfBusy
from zeta.dl.initializers import InitializeWeights as init
from zeta.dl.objectives import ObjectiveFunction as objective
from zeta.dl.optimizers import OptimizationFunction as optimize
from zeta.dl.activations import ActivationFunction as activation

from ..regularizers import RegularizationFunction as regularize


class LogisticRegression:

    def __init__(self, epochs,
                       loss = 'binary-cross-entropy',
                       init_method = 'he-normal',
                       optimizer = {},
                       penalty = 'lasso',
                       penalty_weight = 0,
                       l1_ratio = 0.5):
        self.epochs = epochs
        self.loss = objective(loss)
        self.init_method = init(init_method)
        self.optimizer = optimize(optimizer)
        self.activate = activation('sigmoid')
        self.regularization = regularize(penalty, penalty_weight, l1_ratio = l1_ratio)

    @LogIfBusy
    def fit(self, inputs, targets, verbose = True):
        fit_stats = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
        self.weights = self.init_method.initialize_weights((inputs.shape[1], ))

        for i in range(self.epochs):
            predictions = self.activate._forward(inputs.dot(self.weights))
            cost = self.loss._forward(np.expand_dims(predictions, axis = 1), np.expand_dims(targets, axis = 1)) + self.regularization._regulate(self.weights)
            acc = self.loss._accuracy(predictions, targets)

            fit_stats["train_loss"].append(np.mean(cost))
            fit_stats["train_acc"].append(np.mean(acc))

            if verbose:
                print('TRAINING: Epoch-{} loss: {:.2f} acc: {:.2f}'.format(i+1, cost, acc))

            cost_gradient = self.loss._backward(predictions, targets)
            d_weights = inputs.T.dot(cost_gradient) + self.regularization._derivative(self.weights)
            self.weights = self.optimizer._update(self.weights, d_weights)

        return fit_stats

    @LogIfBusy
    def fit_NR(self, inputs, targets, verbose = True):
        ''' Newton-Raphson Method '''
        fit_stats = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
        self.weights = self.init_method.initialize_weights((inputs.shape[1], ))

        for i in range(self.epochs):
            predictions = self.activate._forward(inputs.dot(self.weights))
            cost = self.loss._forward(np.expand_dims(predictions, axis = 1), np.expand_dims(targets, axis = 1)) + self.regularization._regulate(self.weights)
            acc = self.loss._accuracy(predictions, targets)

            fit_stats["train_loss"].append(np.mean(cost))
            fit_stats["train_acc"].append(np.mean(acc))

            if verbose:
                print('TRAINING: Epoch-{} loss: {:.2f} acc: {:.2f}'.format(i+1, cost, acc))

            diag_grad = np.diag(self.activate._backward(inputs.dot(self.weights)))
            # self.weights += np.linalg.pinv(inputs.T.dot(diag_grad).dot(inputs) + self.regularization._derivative(self.weights)).dot(inputs.T).dot((targets - predictions))
            self.weights += np.linalg.pinv(inputs.T.dot(diag_grad).dot(inputs) + self.regularization._derivative(self.weights)).dot(inputs.T.dot(diag_grad)).dot((targets - predictions))

        return fit_stats

    def predict(self, inputs):
        return np.round(self.activate._forward(inputs.dot(self.weights))).astype(int)
