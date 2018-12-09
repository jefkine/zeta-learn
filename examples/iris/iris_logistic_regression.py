# -*- coding: utf-8 -*-

from datasets.iris import fetch_iris

from ztlearn.utils import *
from ztlearn.optimizers import register_opt
from ztlearn.ml.regression import LogisticRegression

data        = fetch_iris()
input_data  = normalize(data.data[data.target != 2])
input_label = data.target[data.target != 2]

train_data, test_data, train_label, test_label = train_test_split(input_data,
                                                                  input_label,
                                                                  test_size = 0.33, random_seed = 15)

# optimizer definition
opt = register_opt(optimizer_name = 'sgd', momentum = 0.01, learning_rate = 0.01)

# model definition
model     = LogisticRegression(epochs = 1500, optimizer = opt)
fit_stats = model.fit(train_data, train_label)

# fit_stats = model.fit_NR(train_data, train_label) # --- Newton-Raphson Method

print_results(model.predict(test_data), test_label)
plot_metric('accuracy_loss',
                             len(fit_stats["train_loss"]),
                             fit_stats['train_acc'],
                             fit_stats['train_loss'],
                             model_name = 'iris_logistic_regression', legend = ['acc', 'loss'])
