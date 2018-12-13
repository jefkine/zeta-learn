# -*- coding: utf-8 -*-

from ztlearn.utils import *
from ztlearn.optimizers import register_opt
from ztlearn.ml.classification import Perceptron
from ztlearn.datasets.fashion import fetch_fashion_mnist

fashion_mnist = fetch_fashion_mnist()
train_data, test_data, train_label, test_label = train_test_split(normalize(fashion_mnist.data.astype('float32')),
                                                                  one_hot(fashion_mnist.target.astype('int')),
                                                                  test_size   = 0.3,
                                                                  random_seed = 15,
                                                                  cut_off     = 2000)

# plot samples of training data
plot_img_samples(train_data[:40], unhot(train_label[:40]), dataset = 'mnist')

# optimizer definition
opt = register_opt(optimizer_name = 'adam', momentum = 0.01, learning_rate = 0.001)

# model definition
model       = Perceptron(epochs = 600, activation = 'relu', loss = 'cce', init_method = 'he_normal', optimizer =  opt)
fit_stats   = model.fit(train_data, train_label)

predictions = unhot(model.predict(test_data))
print_results(predictions, unhot(test_label))
plot_img_results(test_data[:40], unhot(test_label[:40]), predictions, dataset = 'mnist')
plot_metric('accuracy_loss',
                             len(fit_stats["train_loss"]),
                             fit_stats['train_acc'],
                             fit_stats['train_loss'],
                             model_name = 'fashion_mnist_perceptron',
                             legend     = ['acc', 'loss'])
