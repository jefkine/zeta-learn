# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_mldata

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.optimizers import register_opt
from ztlearn.dl.layers import BatchNormalization, Conv2D
from ztlearn.dl.layers import Dropout, Dense, Flatten, MaxPooling2D


mnist = fetch_mldata('MNIST original')
train_data, test_data, train_label, test_label = train_test_split(mnist.data,
                                                                  mnist.target.astype('int'),
                                                                  test_size = 0.33,
                                                                  random_seed = 5)

plot_img_samples(train_data[:40], train_label[:40], dataset = 'mnist')

opt = register_opt(optimizer_name = 'adam', momentum = 0.01, learning_rate = 0.001)

model = Sequential(init_method = 'he_uniform')
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (1,28,28), padding = 'same'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(10, activation = 'softmax')) # 10 digits classes
model.compile(loss = 'categorical_crossentropy', optimizer = opt)

model_epochs = 12
fit_stats = model.fit(train_data.reshape(-1,1,28,28),
                      one_hot(train_label),
                      batch_size = 128,
                      epochs = model_epochs,
                      validation_data = (test_data.reshape(-1,1,28,28), one_hot(test_label)),
                      shuffle_data = True)

predictions = unhot(model.predict(test_data.reshape(-1,1,28,28), True))

print_results(predictions, test_label)
plot_img_results(test_data[:40], test_label[:40], predictions, dataset = 'mnist')

model_name = 'mnist_cnn'
plot_metric('loss', model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'], model_name = model_name)
plot_metric('accuracy', model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'], model_name = model_name)
