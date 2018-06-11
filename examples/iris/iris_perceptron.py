# -*- coding: utf-8 -*-

from sklearn import datasets

from ztlearn.utils import *
from ztlearn.optimizers import register_opt
from ztlearn.ml.classification import Perceptron

data = datasets.load_iris()
train_data, test_data, train_label, test_label = train_test_split(data.data, one_hot(data.target), test_size = 0.33, random_seed = 5)

opt       = register_opt(optimizer_name = 'adam', momentum = 0.01, learning_rate = 0.001)
model     = Perceptron(epochs = 500, activation = 'leaky_relu', loss = 'cce', init_method = 'he_normal', optimizer = opt)
fit_stats = model.fit(train_data, train_label)

print_results(unhot(model.predict(test_data)), unhot(test_label))
plot_metric('accuracy_loss', len(fit_stats["train_loss"]),
                                                           fit_stats['train_acc'],
                                                           fit_stats['train_loss'],
                                                           model_name = 'iris_perceptron', legend = ['acc', 'loss'])
