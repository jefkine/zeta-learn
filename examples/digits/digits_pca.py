# -*- coding: utf-8 -*-

from sklearn import datasets

from ztlearn.utils import plot_pca
from ztlearn.ml.decomposition import PCA

data = datasets.load_digits()

pca        = PCA(n_components = 2)
components = pca.fit_transform(data.data)

plot_pca(components, n_components = 2, colour_array = data.target, model_name = 'Digits PCA')
