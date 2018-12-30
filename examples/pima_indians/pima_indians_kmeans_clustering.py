# -*- coding: utf-8 -*-

from ztlearn.utils import plot_kmeans
from ztlearn.ml.clustering import KMeans
from ztlearn.datasets.pima import fetch_pima_indians


# fetch dataset
data = fetch_pima_indians()

# model definition
model = KMeans(n_clusters = 2, max_iter = 1000)
centroids = model.fit(data.data[:,[2,5]])

# plot clusters and centroids
plot_kmeans(data.data[:,[2,5]], data.target, centroids, model_clusters = 2)
