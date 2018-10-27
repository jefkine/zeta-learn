# -*- coding: utf-8 -*-

import numpy as np


class PCA(object):

    def __init__(self, n_components = 2):
        self.n_components = n_components

    def fit(self, inputs):
        self.n_samples, self.n_features = np.shape(inputs)

        self.mean = np.mean(inputs, axis = 0)
        inputs   -= self.mean

        self.U, self.S, self.V = np.linalg.svd(inputs, full_matrices = False)

        self.components          = self.V[:self.n_components]
        components_variance      = np.divide(np.square(self.S), (self.n_samples - 1))
        self.components_variance = components_variance[:self.n_components]

        return self

    def fit_transform(self, inputs):
        self.n_samples, self.n_features = np.shape(inputs)

        self.mean = np.mean(inputs, axis = 0)
        inputs   -= self.mean

        U, S, V = np.linalg.svd(inputs, full_matrices = False)

        self.components     = V[:self.n_components]
        transformed_inputs  = U[:, :self.n_components]
        transformed_inputs *= S[:self.n_components]

        components_variance      = np.divide(np.square(S), (self.n_samples - 1))
        self.components_variance = components_variance[:self.n_components]

        return transformed_inputs

    @property
    def transform(self):
        transformed_inputs  = self.U[:, :self.n_components]
        transformed_inputs *= self.S[:self.n_components]

        return transformed_inputs

    def inverse_transform(transformed_inputs):
        return np.dot(transformed_inputs, self.components) + self.mean
