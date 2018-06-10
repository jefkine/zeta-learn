# -*- coding: utf-8 -*-

import numpy as np

class ELU:

    """
    **Exponential Linear Units (ELUs)**

    ELUs are exponential functions which have negative values that allow them
    to push mean unit activations closer to zero like batch normalization but
    with lower computational complexity.

    References:
        [1] Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
            * [Djork-Arné Clevert et. al., 2016] https://arxiv.org/abs/1511.07289
            * [PDF] https://arxiv.org/pdf/1511.07289.pdf

    Args:
        alpha (float32): controls the value to which an ELU saturates for negative net inputs
    """

    def __init__(self, activation_dict):
        self.alpha = activation_dict['alpha'] if 'alpha' in activation_dict else 0.1

    def activation(self, input_signal):

        """
        ELU activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the ELU function applied to the input
        """

        return np.where(input_signal >= 0.0, input_signal, np.multiply(np.expm1(input_signal), self.alpha))

    def derivative(self, input_signal):

        """
        ELU derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the ELU derivative applied to the input
        """

        return np.where(input_signal >= 0.0, 1, self.activation(input_signal) + self.alpha)

    @property
    def activation_name(self):
        return self.__class__.__name__


class SELU:

    """
    **Scaled Exponential Linear Units (SELUs)**

    SELUs are activations which induce self-normalizing properties and are used
    in Self-Normalizing Neural Networks (SNNs). SNNs enable high-level abstract
    representations that tend to automatically converge towards zero mean and
    unit variance.

    References:
        [1] Self-Normalizing Neural Networks (SELUs)
            * [Klambauer, G., et. al., 2017] https://arxiv.org/abs/1706.02515
            * [PDF] https://arxiv.org/pdf/1706.02515.pdf
    Args:
        ALPHA (float32): 1.6732632423543772848170429916717
        _LAMBDA (float32): 1.6732632423543772848170429916717
    """

    ALPHA   = 1.6732632423543772848170429916717
    _LAMBDA = 1.6732632423543772848170429916717

    def __init__(self, activation_dict): pass

    def activation(self, input_signal):

        """
        SELU activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the SELU function applied to the input
        """

        return SELU._LAMBDA * np.where(input_signal >= 0.0, input_signal,
                                                            np.multiply(SELU.ALPHA,
                                                            np.exp(input_signal)) - SELU.ALPHA)

    def derivative(self, input_signal):

        """
        SELU derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the SELU derivative applied to the input
        """

        return SELU._LAMBDA * np.where(input_signal >= 0.0, 1.0, np.multiply(np.exp(input_signal), SELU.ALPHA))

    @property
    def activation_name(self):
        return self.__class__.__name__


class ReLU:

    """
    **Rectified Linear Units (ReLUs)**

    Rectifying neurons are an even better model of biological neurons yielding
    equal or better performance than hyperbolic tangent networks in-spite of
    the hard non-linearity and non-differentiability at zero hence creating
    sparse representations with true zeros which seem remarkably suitable
    for naturally sparse data.

    References:
        [1] Deep Sparse Rectifier Neural Networks
            * [Xavier Glorot., et. al., 2011] http://proceedings.mlr.press/v15/glorot11a.html
            * [PDF] http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf

        [2] Delving Deep into Rectifiers
            * [Kaiming He, et. al., 2015] https://arxiv.org/abs/1502.01852
            * [PDF] https://arxiv.org/pdf/1502.01852.pdf
    """

    def __init__(self, activation_dict): pass

    def activation(self, input_signal):

        """
        ReLU activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the ReLU function applied to the input
        """

        return np.where(input_signal >= 0.0, input_signal, 0.0)

    def derivative(self, input_signal):

        """
        ReLU derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the ReLU derivative applied to the input
        """

        return np.where(input_signal >= 0.0, 1.0, 0.0)

    @property
    def activation_name(self):
        return self.__class__.__name__


class TanH:

    """
    **Tangent Hyperbolic (TanH)**

    The Tangent Hyperbolic function is a rescaled version of the  sigmoid
    function that produces outputs in scale of [-1, +1]. As an activation
    function it produces an output for every input value hence making
    it a continuous function.

    References:
        [1] Hyperbolic Functions
            * [Mathematics Education Centre] https://goo.gl/4Dkkrd
            * [PDF] https://goo.gl/xPSnif
    """

    def __init__(self, activation_dict): pass

    def activation(self, input_signal):

        """
        TanH activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the TanH function applied to the input
        """

        return np.tanh(input_signal)

    def derivative(self, input_signal):

        """
        TanH derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the TanH derivative applied to the input
        """

        return 1 - np.power(self.activation(input_signal), 2)

    @property
    def activation_name(self):
        return self.__class__.__name__


class Sigmoid:

    """
    **Sigmoid Activation Function**

    A Sigmoid function is often used as the output activation function for binary
    classification problems as it outputs values that are in the range (0, 1).
    Sigmoid functions are real-valued and differentiable, producing a curve
    that is 'S-shaped' and feature one local minimum, and one local maximum

    References:
        [1] The influence of the sigmoid function parameters on the speed of
            backpropagation learning https://goo.gl/MavJjj
    """

    def __init__(self, activation_dict): pass

    def activation(self, input_signal):

        """
        Sigmoid activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the Sigmoid function applied to the input
        """

        return np.exp(-np.logaddexp(0, -input_signal))

    def derivative(self, input_signal):

        """
        Sigmoid derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the Sigmoid derivative applied to the input
        """

        output_signal = self.activation(input_signal)
        return np.multiply(output_signal, 1 - output_signal)

    @property
    def activation_name(self):
        return self.__class__.__name__


class SoftPlus:

    """
    **SoftPlus Activation Function**

    A Softplus function is a smooth approximation to the rectifier linear units
    (ReLUs). Near point 0, it is smooth and differentiable and produces outputs
    in scale of (0, +inf).

    References:
        [1] Incorporating Second-Order Functional Knowledge for Better Option Pricing
            * [Charles Dugas, et. al., 2001] https://goo.gl/z3jeYc
            * [PDF] https://goo.gl/z3jeYc
    """

    def __init__(self, activation_dict): pass

    def activation(self, input_signal):

        """
        SoftPlus activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the SoftPlus function applied to the input
        """

        return np.logaddexp(0, input_signal)

    def derivative(self, input_signal):

        """
        SoftPlus derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the SoftPlus derivative applied to the input
        """

        return Sigmoid().activation(input_signal)

    @property
    def activation_name(self):
        return self.__class__.__name__


class Softmax:

    """
    **Softmax Activation Function**

    The Softmax Activation Function is a generalization of the logistic function
    that squashes the outputs of each unit to real values in the range [0, 1]
    but it also divides each output such that the total sum of the outputs
    is equal to 1.

    References:
        [1] Softmax Regression
            [UFLDL Tutorial] https://goo.gl/1qgqdg

        [2] Deep Learning using Linear Support Vector Machines
            * [Yichuan Tang, 2015] https://arxiv.org/abs/1306.0239
            * [PDF] https://arxiv.org/pdf/1306.0239.pdf

        [3] Probabilistic Interpretation of Feedforward Network Outputs
            [Mario Costa, 1989] [PDF] https://goo.gl/ZhBY4r
    """

    def __init__(self, activation_dict): pass

    def activation(self, input_signal):

        """
        Softmax activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the Softmax function applied to the input
        """

        probs = np.exp(input_signal - np.max(input_signal, axis = -1, keepdims = True))
        return probs / np.sum(probs, axis = -1, keepdims = True)

    def derivative(self, input_signal):

        """
        Softmax derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the Softmax derivative applied to the input
        """

        output_signal = self.activation(input_signal)
        return np.multiply(output_signal, 1 - output_signal)

    @property
    def activation_name(self):
        return self.__class__.__name__


class LeakyReLU:

    """
    **LeakyReLU Activation Functions**

    Leaky ReLUs allow a small, non-zero gradient to propagate through the network
    when the unit is not active hence avoiding bottlenecks that can prevent
    learning in the Neural Network.

    References:
        [1] Rectifier Nonlinearities Improve Neural Network Acoustic Models
            * [Andrew L. Mass, et. al., 2013] https://goo.gl/k9fhEZ
            * [PDF] https://goo.gl/v48yXT

        [2] Empirical Evaluation of Rectified Activations in Convolutional Network
            * [Bing Xu, et. al., 2015] https://arxiv.org/abs/1505.00853
            * [PDF] https://arxiv.org/pdf/1505.00853.pdf

    Args:
        alpha (float32): provides for a small non-zero gradient (e.g. 0.01) when the unit is not active.
    """

    def __init__(self, activation_dict):
        self.alpha = activation_dict['alpha'] if 'alpha' in activation_dict else 0.01

    def activation(self, input_signal):

        """
        LeakyReLU activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the LeakyReLU function applied to the input
        """

        return np.where(input_signal >= 0, input_signal, np.multiply(input_signal, self.alpha))

    def derivative(self, input_signal):

        """
        LeakyReLU derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the LeakyReLU derivative applied to the input
        """

        return np.where(input_signal >= 0, 1, self.alpha)

    @property
    def activation_name(self):
        return self.__class__.__name__


class ElliotSigmoid:

    """
    **Elliot Sigmoid Activation Function**

    Elliot Sigmoid squashes each element of the input from the interval [-inf, inf]
    to the interval [-1, 1] with an 'S-shaped' function. The fucntion is fast to
    calculate on simple computing hardware as it does not require any
    exponential or trigonometric functions

    References:
        [1] A better Activation Function for Artificial Neural Networks
            * [David L. Elliott, et. al., 1993] https://goo.gl/qqBdne
            * [PDF] https://goo.gl/fPLPcr
    """

    def __init__(self, activation_dict): pass

    def activation(self, input_signal):

        """
        ElliotSigmoid activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the ElliotSigmoid function applied to the input
        """

        return np.multiply(input_signal, 0.5) / (1 + np.abs(input_signal)) + 0.5

    def derivative(self, input_signal):

        """
        ElliotSigmoid derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the ElliotSigmoid derivative applied to the input
        """

        return 0.5 / np.power((1 + np.abs(input_signal)), 2)

    @property
    def activation_name(self):
        return self.__class__.__name__


class Linear:

    """
    **Linear Activation Function**

    Linear Activation applies identity operation on your data such that the output
    data is proportional to the input data. The function always returns the same
    value that was used as its argument.

    References:
        [1] Identity Function
            [Wikipedia Article] https://en.wikipedia.org/wiki/Identity_function
    """

    def __init__(self, activation_dict): pass

    def activation(self, input_signal):

        """
        Linear activation applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the Linear function applied to the input
        """

        return input_signal

    def derivative(self, input_signal):

        """
        Linear derivative applied to input provided

        Args:
            input_signal (numpy.array): the input numpy array

        Returns:
            numpy.array: the output of the Linear derivative applied to the input
        """

        return input_signal

    @property
    def activation_name(self):
        return self.__class__.__name__


class ActivationFunction:

    _functions = {
        'elu': ELU,
        'selu': SELU,
        'relu': ReLU,
        'tanh': TanH,
        'linear': Linear,
        'identity': Linear,
        'sigmoid': Sigmoid,
        'softmax': Softmax,
        'softplus': SoftPlus,
        'leaky_relu': LeakyReLU,
        'elliot_sigmoid': ElliotSigmoid
    }

    def __init__(self, name, activation_dict = {}):
        if name not in self._functions.keys():
            raise Exception('Activation function must be either one of the following: {}.'.format(', '.join(self._functions.keys())))
        self.activation_func = self._functions[name](activation_dict)

    @property
    def name(self):
        return self.activation_func.activation_name

    def forward(self, input_signal):
        return self.activation_func.activation(input_signal) # returns tuples

    def backward(self, input_signal):
        return self.activation_func.derivative(input_signal)
