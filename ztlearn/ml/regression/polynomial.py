# -*- coding: utf-8 -*-

from numba import jit
from numba import config
from ztlearn.utils import CACHE_FLAG
from ztlearn.utils import NOGIL_FLAG
from ztlearn.utils import DISABLE_JIT_FLAG

config.DISABLE_JIT = DISABLE_JIT_FLAG

from .base import Regression
from ztlearn.utils import normalize
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression(Regression):

    def __init__(self,
                       degree         = 2,
                       epochs         = 100,
                       loss           = 'mean_squared_error',
                       init_method    = 'random_normal',
                       optimizer      = {},
                       penalty        = 'ridge',
                       penalty_weight = 0.5,
                       l1_ratio       = 0.5):

        self.degree = degree
        super(PolynomialRegression, self).__init__(epochs         = epochs,
                                                   loss           = loss,
                                                   init_method    = init_method,
                                                   optimizer      = optimizer,
                                                   penalty        = penalty,
                                                   penalty_weight = penalty_weight,
                                                   l1_ratio       = l1_ratio)

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def fit(self, inputs, targets, verbose = False, normalized = True):
        polynomial_inputs = PolynomialFeatures(degree = self.degree).fit_transform(inputs)

        if normalized:
            polynomial_inputs = normalize(polynomial_inputs)

        fit_stats = super(PolynomialRegression, self).fit(polynomial_inputs, targets, verbose)

        return fit_stats

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def predict(self, inputs, normalized = True):
        polynomial_inputs = PolynomialFeatures(degree = self.degree).fit_transform(inputs)

        if normalized:
            polynomial_inputs = normalize(polynomial_inputs)

        return super(PolynomialRegression, self).predict(polynomial_inputs)
