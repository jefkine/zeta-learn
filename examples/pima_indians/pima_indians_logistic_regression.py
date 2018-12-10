# -*- coding: utf-8 -*-

import numpy as np

from ztlearn.utils import *
from ztlearn.optimizers import register_opt
from ztlearn.ml.regression import LogisticRegression
from ztlearn.datasets.pima import fetch_pima_indians

data = fetch_pima_indians()
# data.target -> using all the features (e.g to use only one feature data.target [:, 5:6])
train_data, test_data, train_label, test_label = train_test_split(z_score(data.data),
                                                                  data.target,
                                                                  test_size   = 0.2,
                                                                  random_seed = 2)

# optimizer definition
opt = register_opt(optimizer_name = 'adam', momentum = 0.01, learning_rate = 0.01)

# model definition
model     = LogisticRegression(epochs = 10000, optimizer = opt)
fit_stats = model.fit(train_data, train_label)

# fit_stats = model.fit_NR(train_data, train_label) # --- Newton-Raphson Method

print_results(model.predict(test_data), np.round(test_label).astype(int))
plot_metric('accuracy_loss',
                             len(fit_stats["train_loss"]),
                             fit_stats['train_acc'],
                             fit_stats['train_loss'],
                             model_name = 'diabetes_logistic_regression',
                             legend     = ['acc', 'loss'])
