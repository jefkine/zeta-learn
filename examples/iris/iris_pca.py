# -*- coding: utf-8 -*-

from ztlearn.utils import plot_pca
from ztlearn.ml.decomposition import PCA
from ztlearn.datasets.iris import fetch_iris

# fetch dataset
data = fetch_iris()

pca        = PCA(n_components = 2)
components = pca.fit_transform(data.data.astype('float64'))

plot_pca(components, n_components = 2, colour_array = data.target.astype('int'), model_name = 'IRIS PCA')
