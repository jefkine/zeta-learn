# -*- coding: utf-8 -*-

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.optimizers import register_opt
from ztlearn.datasets.digits import fetch_digits
from ztlearn.dl.layers import RNN, Dense, Flatten

data = fetch_digits()
train_data, test_data, train_label, test_label = train_test_split(data.data,
                                                                  data.target,
                                                                  test_size   = 0.4,
                                                                  random_seed = 5)

# plot samples of training data
plot_img_samples(train_data, train_label)

# model definition
model = Sequential()
model.add(RNN(128, activation = 'tanh', bptt_truncate = 5, input_shape = (8, 8)))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax')) # 10 digits classes
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
# NOTE: in model.compile you could use a string to define the optimization type

model.summary(model_name = 'digits rnn')

model_epochs = 200
fit_stats = model.fit(train_data.reshape(-1, 8, 8),
                      one_hot(train_label),
                      batch_size      = 128,
                      epochs          = model_epochs,
                      validation_data = (test_data.reshape(-1, 8, 8), one_hot(test_label)),
                      shuffle_data    = True)

predictions = unhot(model.predict(test_data.reshape(-1, 8, 8), True))
print_results(predictions, test_label)
plot_img_results(test_data, test_label, predictions)

plot_metric('loss', model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'], model_name = model.model_name)
plot_metric('accuracy', model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'], model_name = model.model_name)
