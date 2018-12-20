# -*- coding: utf-8 -*-

from ztlearn.utils import z_score
from ztlearn.utils import plot_pca
from ztlearn.ml.decomposition import PCA
from ztlearn.datasets.cifar import fetch_cifar_10

data = fetch_cifar_10()
reshaped_image_dims = 3 * 32 * 32 # ==> (channels * height * width)
reshaped_data       = z_score(data.data.reshape(-1, reshaped_image_dims).astype('float32'))

pca        = PCA(n_components = 2)
components = pca.fit_transform(reshaped_data)

plot_pca(components, n_components = 2, colour_array = data.target, model_name = 'CIFAR-10 PCA')
