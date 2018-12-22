# -*- coding: utf-8 -*-

import numpy as np

from ztlearn.utils import *
from ztlearn.optimizers import register_opt
from ztlearn.datasets.boston import fetch_boston
from ztlearn.ml.regression import LinearRegression
from ztlearn.objectives import ObjectiveFunction as objective

data = fetch_boston()

input_data  = z_score(data.data[:,[5]]) # normalize and work with only one of the features
input_label = data.target

train_data, test_data, train_label, test_label = train_test_split(input_data,
                                                                  input_label,
                                                                  test_size = 0.3)

opt = register_opt(optimizer_name = 'sgd', momentum = 0.01, learning_rate = 0.001)

# model definition
model     = LinearRegression(epochs = 100, optimizer = opt, penalty = 'l1', penalty_weight = 0.8)
fit_stats = model.fit(train_data, train_label)

# fit_stats = model.fit_OLS(train_data, train_label) # ---- Ordinary Least Squares Method

targets     = np.expand_dims(test_label, axis = 1)
predictions = np.expand_dims(model.predict(test_data), axis = 1)
mse         = objective('mean_squared_error').forward(predictions, targets)

print('Mean Squared Error: {:.2f}'.format(mse))

model_name = 'boston_linear_regression'
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
                                    mse, 'Linear Regression',                                    
                                    'Median House Price',
                                    'Average Number of Rooms',
                                    model_name = model_name)
