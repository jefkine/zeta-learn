# -*- coding: utf-8 -*-

from sklearn import datasets

from ztlearn.utils import *
from ztlearn.optimizers import register_opt
from ztlearn.ml.classification import Perceptron

data = datasets.load_digits()
train_data, test_data, train_label, test_label = train_test_split(normalize(data.data),
                                                                  one_hot(data.target),
                                                                  test_size = 0.3, random_seed = 5)

# plot samples of training data
plot_img_samples(train_data, unhot(train_label))

# optimizer definition
opt = register_opt(optimizer_name = 'sgd_momentum', momentum = 0.01, learning_rate = 0.001)

# model definition
model     = Perceptron(epochs = 300, activation = 'selu', loss = 'cce', init_method = 'he_normal', optimizer = opt)
fit_stats = model.fit(train_data, train_label)

predictions = unhot(model.predict(test_data))
print_results(predictions, unhot(test_label))

plot_img_results(test_data, unhot(test_label), predictions)
plot_metric('accuracy_loss',
                             len(fit_stats["train_loss"]),
                             fit_stats['train_acc'],
                             fit_stats['train_loss'],
                             model_name = 'digits_perceptron', legend = ['acc', 'loss'])
