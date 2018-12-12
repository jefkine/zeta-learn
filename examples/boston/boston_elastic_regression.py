# -*- coding: utf-8 -*-

import numpy as np

from ztlearn.utils import *
from ztlearn.optimizers import register_opt
from ztlearn.datasets.boston import fetch_boston
from ztlearn.ml.regression import ElasticNetRegression
from ztlearn.objectives import ObjectiveFunction as objective

data = fetch_boston()

# take the boston data
input_data  = z_score(data.data[:,[5]]) # work with only one of the features: RM
input_label = data.target

train_data, test_data, train_label, test_label = train_test_split(input_data,
                                                                  input_label,
                                                                  test_size = 0.3)

# optimizer definition
opt = register_opt(optimizer_name = 'sgd', momentum = 0.01, learning_rate = 0.001)

# model definition
model = ElasticNetRegression(degree = 3,
                                         epochs         = 100,
                                         optimizer      = opt,
                                         penalty        = 'elastic',
                                         penalty_weight = 0.01,
                                         l1_ratio       = 0.5)

fit_stats   = model.fit(train_data, train_label)
targets     = np.expand_dims(test_label, axis = 1)
predictions = np.expand_dims(model.predict(test_data), axis = 1)
mse         = objective('mean_squared_error').forward(predictions, targets)

print('Mean Squared Error: {:.2f}'.format(mse))

model_name = 'boston_elastic_regression'
plot_metric('accuracy_loss',
                             len(fit_stats['train_loss']),
                             fit_stats['train_acc'],
                             fit_stats['train_loss'],
                             model_name = model_name,
                             legend     = ['acc', 'loss'])

plot_regression_results(train_data,
                                    train_label,
                                    test_data,
                                    test_label,
                                    input_data,
                                    model.predict(input_data),
                                    mse,
                                    'Elastic Regression',
                                    'Median House Price',
                                    'Average Number of Rooms',
                                    model_name = model_name)
