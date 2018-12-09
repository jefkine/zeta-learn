# -*- coding: utf-8 -*-

from datasets.iris import fetch_iris

from ztlearn.utils import *
from ztlearn.optimizers import register_opt
from ztlearn.ml.classification import Perceptron

data = fetch_iris()
train_data, test_data, train_label, test_label = train_test_split(data.data,
                                                                  one_hot(data.target),
                                                                  test_size = 0.3, random_seed = 5)

# optimizer definition
opt = register_opt(optimizer_name = 'sgd_momentum', momentum = 0.01, learning_rate = 0.001)

# model definition
model     = Perceptron(epochs = 500, activation = 'softmax', loss = 'cce', init_method = 'he_normal', optimizer = opt)
fit_stats = model.fit(train_data, train_label)

print_results(unhot(model.predict(test_data)), unhot(test_label))
plot_metric('accuracy_loss',
                             len(fit_stats["train_loss"]),
                             fit_stats['train_acc'],
                             fit_stats['train_loss'],
                             model_name = 'iris_perceptron', legend = ['acc', 'loss'])
