# -*- coding: utf-8 -*-

from sklearn import datasets

from ztlearn.utils import unhot
from ztlearn.utils import one_hot
from ztlearn.utils import plot_loss
from ztlearn.utils import plot_accuracy
from ztlearn.utils import print_results
from ztlearn.utils import train_test_split
from ztlearn.utils import plot_mnist_img_results
from ztlearn.utils import plot_mnist_img_samples

from ztlearn.dl.models import Sequential
from ztlearn.dl.optimizers import register_opt
from ztlearn.dl.layers import LSTM, Dense, Flatten

data = datasets.load_digits()
plot_mnist_img_samples(data)

train_data, test_data, train_label, test_label = train_test_split(data.data,
                                                                  data.target,
                                                                  test_size = 0.3,
                                                                  random_seed = 15)

opt = register_opt(optimizer_name = 'rmsprop', momentum = 0.001, learning_rate = 0.001)

# Model definition
model = Sequential()
model.add(LSTM(128, activation = "tanh", input_shape = (8, 8)))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax')) # mnist classes
model.compile(loss = 'categorical-cross-entropy', optimizer = opt)

model_epochs = 100
fit_stats = model.fit(train_data.reshape(-1,8,8),
                      one_hot(train_label),
                      batch_size = 128,
                      epochs = model_epochs,
                      validation_data = (test_data.reshape(-1,8,8), one_hot(test_label)),
                      shuffle_data = True)

predictions = unhot(model.predict(test_data.reshape(-1,8,8), True))

print_results(predictions, test_label)
plot_mnist_img_results(test_data, test_label, predictions)

plot_loss(model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'])
plot_accuracy(model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'])
