# -*- coding: utf-8 -*-

from ztlearn.utils import *
from ztlearn.dl.layers import LSTM
from ztlearn.dl.models import Sequential
from ztlearn.dl.optimizers import register_opt


x, y, seq_len = gen_mult_sequence_xtyt(1000, 10, 10)
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.4)

print_seq_samples(train_data, train_label)

opt = register_opt(optimizer_name = 'adagrad', momentum = 0.01, learning_rate = 0.01)

# Model definition
model = Sequential()
model.add(LSTM(10, activation = "tanh", input_shape = (10, seq_len)))
model.compile(loss = 'categorical_crossentropy', optimizer = opt)

model_epochs = 100
fit_stats = model.fit(train_data,
                      train_label,
                      batch_size = 100,
                      epochs = model_epochs,
                      validation_data = (test_data, test_label))

print_seq_results(model.predict(test_data,(0,2,1)), test_label, test_data, unhot_axis = 2)

plot_loss(model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'])
plot_accuracy(model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'])
