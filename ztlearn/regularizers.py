# -*- coding: utf-8 -*-

import numpy as np

from numba import jit, config
from ztlearn.utils import jit_flag
config.NUMBA_DISABLE_JIT = jit_flag

# Note: careful as np.multiply does an elementwise multiply on numpy arrays
#       asterisk (*) does the same but will perfom matrix multiplication on mat (numpy matrices)

class L1Regularization:

    """
    **Lasso Regression (L1Regularization)**

    L1Regularization adds sum of the absolute value magnitudes of parameters as
    penalty term to the loss function

    References:
        [1] Regularization (mathematics)
            [Wikipedia Article] https://en.wikipedia.org/wiki/Regularization_(mathematics)

        [2] Regression shrinkage and selection via the lasso
            * [R Tibshirani, 1996] https://goo.gl/Yh9bBU
            * [PDF] https://goo.gl/mQP5mA

        [3] Feature selection, L1 vs. L2 regularization, and rotational invariance
            [Andrew Y. Ng, ] [PDF] https://goo.gl/rbwNCt

    Args:
        _lambda  (float32): controls the weight of the penalty term
    """

    def __init__(self, _lambda, **kwargs):
        self._lambda = _lambda

    @jit(nogil = True, cache = True)
    def regulate(self, weights):
        return np.multiply(self._lambda, np.linalg.norm(weights))

    @jit(nogil = True, cache = True)
    def derivative(self, weights):
        return np.multiply(self._lambda, np.sign(weights))

    @property
    def regulation_name(self):
        return self.__class__.__name__


class L2Regularization:

    """
    **Lasso Regression (L2Regularization)**

    L1Regularization adds sum of the squared magnitudes of parameters as penalty
    term to the loss function

    References:
        [1] Regularization (mathematics)
            [Wikipedia Article] https://en.wikipedia.org/wiki/Regularization_(mathematics)

        [2] Regression shrinkage and selection via the lasso
            * [R Tibshirani, 1996] https://goo.gl/Yh9bBU
            * [PDF] https://goo.gl/mQP5mA

        [3] Feature selection, L1 vs. L2 regularization, and rotational invariance
            [Andrew Y. Ng, ] [PDF] https://goo.gl/rbwNCt

    Args:
        _lambda (float32): controls the weight of the penalty term
    """

    def __init__(self, _lambda, **kwargs):
        self._lambda = _lambda

    @jit(nogil = True, cache = True)
    def regulate(self, weights):
        return np.multiply(self._lambda, (0.5 *  weights.T.dot(weights)))

    @jit(nogil = True, cache = True)
    def derivative(self, weights):
        return np.multiply(self._lambda, weights)

    @property
    def regulation_name(self):
        return self.__class__.__name__


class ElasticNetRegularization:

    """
    **Elastic Net Regularization (ElasticNetRegularization)**

    ElasticNetRegularization adds both absolute value of magnitude and squared
    magnitude of coefficient as penalty term to the loss function

    References:
        [1] Regularization (mathematics)
            [Wikipedia Article] https://en.wikipedia.org/wiki/Regularization_(mathematics)

    Args:
        _lambda  (float32): controls the weight of the penalty term
        l1_ratio (float32): controls the value l1 penalty as a ratio of total penalty added to the loss function
    """

    def __init__(self, _lambda, l1_ratio):
        self._lambda  = _lambda
        self.l1_ratio = l1_ratio

    @jit(nogil = True, cache = True)
    def regulate(self, weights):
        return np.multiply(self._lambda, (((self.l1_ratio * 0.5) * weights.T.dot(weights)) + ((1 - self.l1_ratio) * np.linalg.norm(weights))))

    @jit(nogil = True, cache = True)
    def derivative(self, weights):
        return np.multiply(self._lambda, (((self.l1_ratio * 0.5) * weights) + ((1 - self.l1_ratio) *  np.sign(weights))))

    @property
    def regulation_name(self):
        return self.__class__.__name__


class RegularizationFunction:

    _regularizers = {
        'l1'          : L1Regularization,
        'l2'          : L2Regularization,
        'lasso'       : L1Regularization,
        'ridge'       : L2Regularization,
        'elastic'     : ElasticNetRegularization,
        'elastic_net' : ElasticNetRegularization
    }

    def __init__(self, name = 'lasso', _lambda = 0.5, l1_ratio = 0.5):
        if name not in self._regularizers.keys():
            raise Exception('Regularization function must be either one of the following: {}.'.format(', '.join(self._regularizers.keys())))
        self.regularization_func = self._regularizers[name](_lambda, l1_ratio = l1_ratio)

    @property
    def name(self):
        return self.regularization_func.regularization_name

    def regulate(self, weights):
        return self.regularization_func.regulate(weights)

    def derivative(self, weights):
        return self.regularization_func.derivative(weights)
