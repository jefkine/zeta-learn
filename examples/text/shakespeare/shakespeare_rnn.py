# -*- coding: utf-8 -*-

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.optimizers import register_opt
from ztlearn.dl.layers import RNN, Flatten, Dense


text = open('../../../ztlearn/datasets/text/tinyshakespeare_short.txt').read().lower()
x, y, len_chars = gen_char_sequence_xtym(text, maxlen = 30, step = 1)
del text

train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.4)

# optimizer definition
opt = register_opt(optimizer_name = 'rmsprop', momentum = 0.1, learning_rate = 0.01)

# model definition
model = Sequential()
model.add(RNN(128, activation = 'tanh', bptt_truncate = 24, input_shape = (30, len_chars)))
model.add(Flatten())
model.add(Dense(len_chars,  activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = opt)

model.summary('shakespeare rnn')

model_epochs = 20
fit_stats = model.fit(train_data,
                      train_label,
                      batch_size      = 128,
                      epochs          = model_epochs,
                      validation_data = (test_data, test_label))

model_name = model.model_name
plot_metric('loss', model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'], model_name = model_name)
plot_metric('accuracy', model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'], model_name = model_name)
