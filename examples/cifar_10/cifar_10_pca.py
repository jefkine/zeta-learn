# -*- coding: utf-8 -*-

from ztlearn.utils import plot_pca
from ztlearn.ml.decomposition import PCA
from ztlearn.datasets.cifar import fetch_cifar_10

data = fetch_cifar_10()
transformed_image_dims = 3 * 32 * 32 # ==> (channels * height * width)
transformed_data       = data.data.reshape(-1, transformed_image_dims).astype('float32') / 255

pca        = PCA(n_components = 2)
components = pca.fit_transform(transformed_data)

plot_pca(components, n_components = 2, colour_array = data.target, model_name = 'CIFAR-10 PCA')
