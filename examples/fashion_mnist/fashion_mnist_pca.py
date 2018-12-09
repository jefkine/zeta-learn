# -*- coding: utf-8 -*-

from datasets.fashion import fetch_fashion_mnist

from ztlearn.utils import plot_pca
from ztlearn.ml.decomposition import PCA

fashion_mnist = fetch_fashion_mnist()

pca        = PCA(n_components = 2)
components = pca.fit_transform(fashion_mnist.data.astype('float64'))

plot_pca(components, n_components = 2, colour_array = fashion_mnist.target.astype('int'), model_name = 'FASHION MNIST PCA')
