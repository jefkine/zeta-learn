# -*- coding: utf-8 -*-

import numpy as np


class Decay(object):

    def __init__(self, lr, decay, epoch, min_lr, max_lr):
        self.lr     = lr
        self.decay  = decay
        self.epoch  = epoch
        self.min_lr = min_lr
        self.max_lr = max_lr

    @property
    def clip_lr(self):
        return np.clip(self.lr, self.min_lr, self.max_lr)


class InverseTimeDecay(Decay):

    def __init__(self, lr, decay, epoch, min_lr, max_lr, step_size):
        super(InverseTimeDecay, self).__init__(lr, decay, epoch, min_lr, max_lr)

    @property
    def decompose(self):
        self.lr *= (1. / (1 + self.decay * self.epoch))

        return super(InverseTimeDecay, self).clip_lr

    @property
    def decay_name(self):
        return self.__class__.__name__


class StepDecay(Decay):
    """ decay the learning rate every after step_size steps """

    def __init__(self, lr, decay, epoch, min_lr, max_lr, step_size):
        super(StepDecay, self).__init__(lr, decay, epoch, min_lr, max_lr)
        self.step_size = step_size

    @property
    def decompose(self):
        self.lr *= np.power(self.decay, ((1 + self.epoch) // self.step_size))

        return super(StepDecay, self).clip_lr

    @property
    def decay_name(self):
        return self.__class__.__name__


class ExponetialDecay(Decay):

    def __init__(self, lr, decay, epoch, min_lr, max_lr, step_size):
        super(ExponetialDecay, self).__init__(lr, decay, epoch, min_lr, max_lr)

    @property
    def decompose(self):
        self.lr *= np.power(self.decay, self.epoch)

        return super(ExponetialDecay, self).clip_lr

    @property
    def decay_name(self):
        return self.__class__.__name__


class NaturalExponentialDecay(Decay):

    def __init__(self, lr, decay, epoch, min_lr, max_lr, step_size):
        super(NaturalExponentialDecay, self).__init__(lr, decay, epoch, min_lr, max_lr)

    @property
    def decompose(self):
        self.lr *= np.exp(-self.decay * self.epoch)

        return super(NaturalExponentialDecay, self).clip_lr

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
                       lr        = 0.001,
                       name      = 'inverse_time_decay',
                       decay     = 1e-6,
                       epoch     = 1,
                       min_lr    = 0.,
                       max_lr    = np.inf,
                       step_size = 10.0):

        if name not in self._functions.keys():
            raise Exception('Decay function must be either one of the following: {}.'.format(', '.join(self._functions.keys())))
        self.decay_func = self._functions[name](lr, decay, epoch, min_lr, max_lr, step_size)

    @property
    def name(self):
        return self.decay_func.decay_name

    @property
    def decompose(self):
        return self.decay_func.decompose
