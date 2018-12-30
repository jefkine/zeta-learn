# -*- coding: utf-8 -*-

from ztlearn.utils import plot_pca
from ztlearn.ml.decomposition import PCA
from ztlearn.datasets.pima import fetch_pima_indians

# fetch dataset
data = fetch_pima_indians()

pca        = PCA(n_components = 2)
components = pca.fit_transform(data.data[:,[2,5]].astype('float64'))

plot_pca(components, n_components = 2, colour_array = data.target.astype('int'), model_name = 'PIMA INDIANS PCA')
