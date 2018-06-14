# -*- coding: utf-8 -*-

import numpy as np

from numba import jit, config
from ztlearn.utils import JIT_FLAG, CACHE_FLAG, NOGIL_FLAG

from .base import Regression
from ztlearn.utils import LogIfBusy

config.NUMBA_DISABLE_JIT = JIT_FLAG

class LinearRegression(Regression):

    def __init__(self,
                       epochs         = 100,
                       loss           = 'mean_squared_error',
                       init_method    = 'random_normal',
                       optimizer      = {},
                       penalty        = 'ridge',
                       penalty_weight = 0.5,
                       l1_ratio       = 0.5):

        super(LinearRegression, self).__init__(epochs         = epochs,
                                               loss           = loss,
                                               init_method    = init_method,
                                               optimizer      = optimizer,
                                               penalty        = penalty,
                                               penalty_weight = penalty_weight,
                                               l1_ratio       = l1_ratio)

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def fit(self, inputs, targets, verbose =  False):
        fit_stats = super(LinearRegression, self).fit(inputs, targets, verbose)

        return fit_stats

    @LogIfBusy
    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def fit_OLS(self, inputs, targets, verbose = True):
        fit_stats    = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
        inputs       = np.column_stack((np.ones(inputs.shape[0]), inputs))
        self.weights = np.linalg.inv(inputs.T.dot(inputs)).dot(inputs.T).dot(targets)

        return fit_stats
