# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_mldata

from ztlearn.utils import plot_pca
from ztlearn.ml.decomposition import PCA

mnist = fetch_mldata('MNIST original')

pca        = PCA(n_components = 2)
components = pca.fit_transform(mnist.data.astype('float64'))

plot_pca(components, n_components = 2, colour_array = mnist.target.astype('int'), model_name = 'MNIST PCA')
