# -*- coding: utf-8 -*-

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.optimizers import register_opt
from ztlearn.dl.layers import RNN, Dense, Flatten
from ztlearn.datasets.cifar import fetch_cifar_100

data = fetch_cifar_100()
train_data, test_data, train_label, test_label = train_test_split(data.data,
                                                                  data.target,
                                                                  test_size   = 0.3,
                                                                  random_seed = 5)

# plot samples of training data
plot_img_samples(train_data, train_label, dataset = 'cifar', channels = 3)

transformed_image_dims = 3 * 1024 # ==> (channels * (height * width))
transformed_train_data = z_score(train_data.reshape(42000, transformed_image_dims).astype('float32'))
transformed_test_data  = z_score(test_data.reshape(18000, transformed_image_dims).astype('float32'))

# optimizer definition
opt = register_opt(optimizer_name = 'adam', momentum = 0.01, learning_rate = 0.0001)

# model definition
model = Sequential()
model.add(RNN(256, activation = 'tanh', bptt_truncate = 5, input_shape = (3, 1024)))
model.add(Flatten())
model.add(Dense(100, activation = 'softmax')) # 10 digits classes
model.compile(loss = 'categorical_crossentropy', optimizer = opt)

model.summary(model_name = 'cifar-100 rnn')

model_epochs = 10 # add more epochs
fit_stats = model.fit(transformed_train_data.reshape(-1, 3, 1024),
                      one_hot(train_label),
                      batch_size      = 128,
                      epochs          = model_epochs,
                      validation_data = (test_data.reshape(-1, 3, 1024), one_hot(test_label)),
                      shuffle_data    = True)

predictions = unhot(model.predict(transformed_test_data.reshape(-1, 3, 1024), True))
print_results(predictions, test_label)
plot_img_results(test_data, test_label, predictions, dataset = 'cifar', channels = 3)

model_name = model.model_name
plot_metric('loss', model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'], model_name = model_name)
plot_metric('accuracy', model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'], model_name = model_name)
