# -*- coding: utf-8 -*-

import numpy as np

# implementation based on techniques as seen here: https://github.com/goldsborough/k-means/blob/master/python/k_means.py
class KMeans:

    def __init__(self, n_clusters = 2, max_iter = 300, random_state = None):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.random_state = random_state

    def _initialize_centroids(self, inputs):
        self.n_samples, self.n_features = np.shape(inputs)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # get random indices for centroid and use them to initialize centroids
        centroid_indices = np.random.choice(range(self.n_samples), self.n_clusters)
        self.centroids   = inputs[centroid_indices].T

        # stack inputs to form a tensor of dim [self.n_samples, self.n_features, self.n_clusters]
        self.stacked_inputs = np.stack([inputs] * self.n_clusters, axis = -1)

        self.all_rows    = np.arange(self.n_samples) # get all row indices in an np array
        self.sparse_data = np.zeros([self.n_samples, self.n_clusters, self.n_features])

    def fit(self, inputs):

        self._initialize_centroids(inputs) # initialize the centroids randomly

        for i in range(self.max_iter):

            # calculate distances of data_points from centroids
            # returns a distance tensor of dim [no of data_points, distance_from_each_centroid]
            distances = np.linalg.norm(self.stacked_inputs - self.centroids, axis = 1)

            # given n centroids the 'distance_from_each_centroid' metric consists of n diffrent distances to the n centroids
            # to find the minimum distance amongst the n diffrent distance we use the np.argmin function on this axis.
            # closest_centroid = np.argmin(distances, axis = -1). closest_centroid is a tensor of dim [no of data_points]
            # closest_centroid tensor consists of a collection of the indices of closest centroids.

            # for each data_point, the [row number, closest_centroid] index is its position in the sparse_data tensor
            # this operation fills in the sparse_data tensor positions at [all_rows, closest_centroid] with data from the inputs
            self.sparse_data[self.all_rows, np.argmin(distances, axis = -1)] = inputs

            # save current centroids for model convergence check
            prior_centroids = self.centroids

            # calculate the mean of all the newly formed clusters
            # get the sum of elements on the first axis (i.e axis = 0)
            # divide by the count of non zero elements in the sparse_data tensor on the first axis (i.e axis = 0)
            # also clip at a minimum of 1 to avoid division by zero
            self.centroids = np.divide(np.sum(self.sparse_data, axis = 0),
                                       np.clip(np.count_nonzero(self.sparse_data, axis = 0), a_min = 1, a_max = None)).T

            # determine if current centroids are diffrent with prior centroids. break if not
            if not np.any(self.centroids - prior_centroids):
                break

        return self.centroids.T
