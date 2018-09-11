# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs

from ztlearn.utils import plot_kmeans
from ztlearn.ml.clustering import KMeans

# generate fake data
data, labels = make_blobs(n_samples = 1000, n_features = 2, centers = 4)

# model definition
model = KMeans(n_clusters = 4, max_iter = 1000)
centroids = model.fit(data)

# plot clusters and centroids
plot_kmeans(data, labels, centroids, model_clusters = 4)
