# -*- coding: utf-8 -*-

from datasets.mnist import fetch_mnist

from ztlearn.utils import plot_pca
from ztlearn.ml.decomposition import PCA

mnist = fetch_mnist()

pca        = PCA(n_components = 2)
components = pca.fit_transform(mnist.data.astype('float64'))

plot_pca(components, n_components = 2, colour_array = mnist.target.astype('int'), model_name = 'MNIST PCA')
