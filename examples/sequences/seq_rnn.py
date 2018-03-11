# -*- coding: utf-8 -*-

from ztlearn.utils import plot_loss
from ztlearn.utils import plot_accuracy
from ztlearn.utils import train_test_split
from ztlearn.utils import print_seq_samples
from ztlearn.utils import print_seq_results
from ztlearn.utils import gen_mult_sequence_xtym

from ztlearn.dl.models import Sequential
from ztlearn.dl.optimizers import register_opt
from ztlearn.dl.layers import RNN, Flatten, Dense


x, y, seq_len = gen_mult_sequence_xtym(3000, 10, 10)
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.3)

print_seq_samples(train_data, train_label, 0)

opt = register_opt(optimizer_name = 'adam', momentum = 0.01, learning_rate = 0.01)

# Model definition
model = Sequential()
model.add(RNN(5, activation = "tanh", bptt_truncate = 5, input_shape = (9, seq_len)))
model.add(Flatten())
model.add(Dense(seq_len, activation = 'softmax'))
model.compile(loss = 'categorical-cross-entropy', optimizer = opt)

model_epochs = 15
fit_stats = model.fit(train_data,
                      train_label,
                      batch_size = 100,
                      epochs = model_epochs,
                      validation_data = (test_data, test_label))

print_seq_results(model.predict(test_data), test_label, test_data)

plot_loss(model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'])
plot_accuracy(model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'])
