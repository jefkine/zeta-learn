# -*- coding: utf-8 -*-

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.optimizers import register_opt
from ztlearn.dl.layers import LSTM, Dense, Flatten
from ztlearn.datasets.fashion import fetch_fashion_mnist

fashion_mnist = fetch_fashion_mnist()
train_data, test_data, train_label, test_label = train_test_split(fashion_mnist.data,
                                                                  fashion_mnist.target.astype('int'),
                                                                  test_size   = 0.3,
                                                                  random_seed = 15,
                                                                  cut_off     = 2000)

# plot samples of training data
plot_img_samples(train_data[:40], train_label[:40], dataset = 'mnist')

# optimizer definition
opt = register_opt(optimizer_name = 'rmsprop', momentum = 0.01, learning_rate = 0.001)

# model definition
model = Sequential()
model.add(LSTM(128, activation = 'tanh', input_shape = (28, 28)))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax')) # 10 digits classes
model.compile(loss = 'categorical_crossentropy', optimizer = opt)

model.summary('fashion mnist lstm')

model_epochs = 100
fit_stats = model.fit(train_data.reshape(-1, 28, 28),
                      one_hot(train_label),
                      batch_size      = 128,
                      epochs          = model_epochs,
                      validation_data = (test_data.reshape(-1, 28, 28), one_hot(test_label)),
                      shuffle_data    = True)

predictions = unhot(model.predict(test_data.reshape(-1, 28, 28), True))
print_results(predictions, test_label)
plot_img_results(test_data[:40], test_label[:40], predictions, dataset = 'mnist')

model_name = model.model_name
plot_metric('loss', model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'], model_name = model_name)
plot_metric('accuracy', model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'], model_name = model_name)
