# -*- coding: utf-8 -*-

from sklearn import datasets

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.dl.optimizers import register_opt
from ztlearn.dl.layers import RNN, Dense, Flatten

data = datasets.load_digits()
plot_digits_img_samples(data)

train_data, test_data, train_label, test_label = train_test_split(data.data,
                                                                  data.target,
                                                                  test_size = 0.4,
                                                                  random_seed = 5)

opt = register_opt(optimizer_name = 'adam', momentum = 0.01, learning_rate = 0.001)

# Model definition
model = Sequential()
model.add(RNN(128, activation = "tanh", bptt_truncate = 5, input_shape = (8, 8)))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax')) # 10 digits classes
model.compile(loss = 'categorical_crossentropy', optimizer = opt)

model_epochs = 100
fit_stats = model.fit(train_data.reshape(-1,8,8),
                      one_hot(train_label),
                      batch_size = 128,
                      epochs = model_epochs,
                      validation_data = (test_data.reshape(-1,8,8), one_hot(test_label)),
                      shuffle_data = True,
                      verbose = False)

predictions = unhot(model.predict(test_data.reshape(-1,8,8), True))

print_results(predictions, test_label)
plot_digits_img_results(test_data, test_label, predictions)
plot_loss(model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'])
plot_accuracy(model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'])
