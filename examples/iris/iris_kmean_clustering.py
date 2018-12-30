# -*- coding: utf-8 -*-

from ztlearn.utils import plot_kmeans
from ztlearn.ml.clustering import KMeans
from ztlearn.datasets.iris import fetch_iris


# fetch dataset
data = fetch_iris()

# model definition
model = KMeans(n_clusters = 3, max_iter = 1000)
centroids = model.fit(data.data)

# plot clusters and centroids
plot_kmeans(data.data, data.target, centroids, model_clusters = 3)
