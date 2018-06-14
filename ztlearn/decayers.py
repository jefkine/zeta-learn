# -*- coding: utf-8 -*-

import numpy as np

from numba import jit, config
from ztlearn.utils import DISABLE_JIT_FLAG, CACHE_FLAG, NOGIL_FLAG

config.NUMBA_DISABLE_JIT = DISABLE_JIT_FLAG

class Decay(object):

    def __init__(self, lrate, decay, epoch, min_lrate, max_lrate):
        self.lrate     = lrate
        self.decay     = decay
        self.epoch     = epoch
        self.min_lrate = min_lrate
        self.max_lrate = max_lrate

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def clip_lrate(self):
        return np.clip(self.lrate, self.min_lrate, self.max_lrate)


class InverseTimeDecay(Decay):

    def __init__(self, lrate, decay, epoch, min_lrate, max_lrate, step_size):
        super(InverseTimeDecay, self).__init__(lrate, decay, epoch, min_lrate, max_lrate)

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def decompose(self):
        self.lrate *= (1. / (1 + self.decay * self.epoch))

        return super(InverseTimeDecay, self).clip_lrate()

    @property
    def decay_name(self):
        return self.__class__.__name__


class StepDecay(Decay):
    """Decay the learning rate every after step_size steps"""

    def __init__(self, lrate, decay, epoch, min_lrate, max_lrate, step_size):
        super(StepDecay, self).__init__(lrate, decay, epoch, min_lrate, max_lrate)
        self.step_size = step_size

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def decompose(self):
        self.lrate *= np.power(self.decay, ((1 + self.epoch) // self.step_size))

        return super(StepDecay, self).clip_lrate()

    @property
    def decay_name(self):
        return self.__class__.__name__


class ExponetialDecay(Decay):

    def __init__(self, lrate, decay, epoch, min_lrate, max_lrate, step_size):
        super(ExponetialDecay, self).__init__(lrate, decay, epoch, min_lrate, max_lrate)

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def decompose(self):
        self.lrate *= np.power(self.decay, self.epoch)

        return super(ExponetialDecay, self).clip_lrate()

    @property
    def decay_name(self):
        return self.__class__.__name__


class NaturalExponentialDecay(Decay):

    def __init__(self, lrate, decay, epoch, min_lrate, max_lrate, step_size):
        super(NaturalExponentialDecay, self).__init__(lrate, decay, epoch, min_lrate, max_lrate)

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def decompose(self):
        self.lrate *= np.exp(-self.decay * self.epoch)

        return super(NaturalExponentialDecay, self).clip_lrate()

    @property
    def decay_name(self):
        return self.__class__.__name__


class DecayFunction:

    _functions = {
        'step_decay'                : StepDecay,
        'exponential_decay'         : ExponetialDecay,
        'inverse_time_decay'        : InverseTimeDecay,
        'natural_exponential_decay' : NaturalExponentialDecay
    }

    def __init__(self,
                       lrate     = 0.001,
                       name      = 'inverse_time_decay',
                       decay     = 1e-6,
                       epoch     = 1,
                       min_lrate = 0.,
                       max_lrate = np.inf,
                       step_size = 10.0):

        if name not in self._functions.keys():
            raise Exception('Decay function must be either one of the following: {}.'.format(', '.join(self._functions.keys())))
        self.decay_func = self._functions[name](lrate, decay, epoch, min_lrate, max_lrate, step_size)

    @property
    def name(self):
        return self.decay_func.decay_name

    @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
    def decompose(self):
        return self.decay_func.decompose()
