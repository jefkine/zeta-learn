# -*- coding: utf-8 -*-

import numpy as np

class Objective(object):

    def clip(self, predictions, epsilon = 1e-15):
        clipped_predictions = np.clip(predictions, epsilon, 1 - epsilon)
        clipped_divisor = np.maximum(predictions * (1 - predictions), epsilon)
        return clipped_predictions, clipped_divisor

    @property
    def objective_name(self):
        return self.__class__.__name__


class MeanSquaredError:

    """
    **Mean Squared error (MSE)**

    MSE measures the average squared difference between the predictions and the
    targets. The closer the predictions are to the targets the more efficient
    the estimator.

    References:
        [1] Mean Squared error
            [Wikipedia Article] https://en.wikipedia.org/wiki/Mean_squared_error
    """

    def loss(self, predictions, targets, np_type):

        """
        Applies the MeanSquaredError Loss to prediction and targets provided

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array

        Returns:
            numpy.array: the output of MeanSquaredError Loss to prediction and targets
        """

        return 0.5 * np.mean(np.sum(np.power(predictions - targets, 2), axis = 1))

    def derivative(self, predictions, targets):

        """
        Applies the MeanSquaredError Derivative to prediction and targets provided

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array

        Returns:
            numpy.array: the output of MeanSquaredError Derivative to prediction and targets
        """

        return predictions - targets

    def accuracy(self, predictions, targets, threshold = 0.5):
        return 0

    @property
    def objective_name(self):
        return self.__class__.__name__


class HellingerDistance:

    """
    **Hellinger Distance**

    Hellinger Distance is used to quantify the similarity between two probability
    distributions.

    References:
        [1] Hellinger Distance
            [Wikipedia Article] https://en.wikipedia.org/wiki/Hellinger_distance
    """

    SQRT_2 = np.sqrt(2)

    def sqrt_difference(self, predictions, targets):
        return np.sqrt(predictions) - np.sqrt(targets)

    def loss(self, predictions, targets, np_type):

        """
        Applies the HellingerDistance Loss to prediction and targets provided

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array

        Returns:
            numpy.array: the output of HellingerDistance Loss to prediction and targets
        """

        root_difference = self.sqrt_difference(predictions, targets)
        return np.mean(np.sum(np.power(root_difference, 2), axis = 1) / HellingerDistance.SQRT_2)

    def derivative(self, predictions, targets):

        """
        Applies the HellingerDistance Derivative to prediction and targets provided

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array

        Returns:
            numpy.array: the output of HellingerDistance Derivative to prediction and targets
        """

        root_difference = self.sqrt_difference(predictions, targets)
        return root_difference / (HellingerDistance.SQRT_2 * np.sqrt(predictions))

    def accuracy(self, predictions, targets, threshold = 0.5):
        return 0

    @property
    def objective_name(self):
        return self.__class__.__name__


class BinaryCrossEntropy(Objective):

    """
    **Binary Cross Entropy**

    Binary CrossEntropy measures the performance of a classification model whose
    output is a probability value between 0 & 1. 'Binary' is meant for discrete
    classification tasks in which the classes are independent and not mutually
    exclusive. Targets here could be either 0 or 1 scalar

    References:
        [1] Cross Entropy
            [Wikipedia Article] https://en.wikipedia.org/wiki/Cross_entropy
    """

    def loss(self, predictions, targets, np_type):

        """
        Applies the BinaryCrossEntropy Loss to prediction and targets provided

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array

        Returns:
            numpy.array: the output of BinaryCrossEntropy Loss to prediction and targets
        """

        clipped_predictions, _ = super(BinaryCrossEntropy, self).clip(predictions)
        return np.mean(-np.sum(targets * np.log( clipped_predictions) + (1 - targets) * np.log(1 - clipped_predictions), axis = 1))

    def derivative(self, predictions, targets):

        """
        Applies the BinaryCrossEntropy Derivative to prediction and targets provided

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array

        Returns:
            numpy.array: the output of BinaryCrossEntropy Derivative to prediction and targets
        """

        clipped_predictions, clipped_divisor = super(BinaryCrossEntropy, self).clip(predictions)
        return (clipped_predictions - targets) / clipped_divisor

    def accuracy(self, predictions, targets, threshold = 0.5):

        """
        Calculates the BinaryCrossEntropy Accuracy Score given prediction and targets

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array
            threshold   (numpy.float32): the threshold value

        Returns:
            numpy.float32: the output of BinaryCrossEntropy Accuracy Score
        """

        return 1 - np.count_nonzero((predictions > threshold) == targets) / float(targets.size)

    @property
    def objective_name(self):
        return self.__class__.__name__


class CategoricalCrossEntropy(Objective):

    """
    **Categorical Cross Entropy**

    Categorical Cross Entropy measures the performance of a classification model
    whose output is a probability value between 0 and 1. 'Categorical' is  meant
    for discrete classification tasks in which the classes are mutually exclusive.

    References:
        [1] Cross Entropy
            [Wikipedia Article] https://en.wikipedia.org/wiki/Cross_entropy
    """

    def loss(self, predictions, targets, np_type):

        """
        Applies the CategoricalCrossEntropy Loss to prediction and targets provided

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array

        Returns:
            numpy.array: the output of CategoricalCrossEntropy Loss to prediction and targets
        """

        clipped_predictions, _ = super(CategoricalCrossEntropy, self).clip(predictions)
        return np.mean(-np.sum(targets * np.log(clipped_predictions), axis = 1))

    def derivative(self, predictions, targets):

        """
        Applies the CategoricalCrossEntropy Derivative to prediction and targets provided

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array

        Returns:
            numpy.array: the output of CategoricalCrossEntropy Derivative to prediction and targets
        """

        clipped_predictions, _ = super(CategoricalCrossEntropy, self).clip(predictions)
        return clipped_predictions - targets

    def accuracy(self, predictions, targets):

        """
        Calculates the CategoricalCrossEntropy Accuracy Score given prediction and targets

        Args:
            predictions (numpy.array): the predictions numpy array
            targets     (numpy.array): the targets numpy array

        Returns:
            numpy.float32: the output of CategoricalCrossEntropy Accuracy Score
        """

        return np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis = 1))

    @property
    def objective_name(self):
        return self.__class__.__name__


class ObjectiveFunction:

    _functions = {
        'mse': MeanSquaredError,
        'mean-squared-error': MeanSquaredError,
        'hellinger-distance': HellingerDistance,
        'binary-cross-entropy': BinaryCrossEntropy,
        'categorical-cross-entropy': CategoricalCrossEntropy
    }

    def __init__(self, name):
        if name not in self._functions.keys():
            raise Exception('Objective function must be either one of the following: {}.'.format(', '.join(self._functions.keys())))
        self.objective_func = self._functions[name]()

    @property
    def name(self):
        return self.objective_func.objective_name

    def _forward(self, predictions, targets, np_type = np.float32):
        return self.objective_func.loss(predictions, targets, np_type)

    def _backward(self, predictions, targets):
        return self.objective_func.derivative(predictions, targets)

    def _accuracy(self, predictions, targets):
        return self.objective_func.accuracy(predictions, targets)
