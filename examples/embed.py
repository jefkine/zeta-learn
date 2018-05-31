# -*- coding: utf-8 -*-

import numpy as np

from ztlearn.dl.layers import Embedding
from ztlearn.dl.models import Sequential
from ztlearn.utils import train_test_split
from ztlearn.optimizers import register_opt


opt = register_opt(optimizer_name = 'sgd_momentum', momentum = 0.01, learning_rate = 0.001)
model = Sequential(init_method = 'he_normal')
model.add(Embedding(10, 2, activation = 'selu', input_shape = (1, 10)))
model.compile(loss = 'categorical_crossentropy', optimizer = opt)

train_data = np.random.randint(10, size=(5, 1, 10))
train_label = np.random.randint(14, size=(5, 1, 10))

train_data, test_data, train_label, test_label = train_test_split(train_data,
                                                                  train_label,
                                                                  test_size = 0.1)

fit_stats = model.fit(train_data, train_label, batch_size = 4, epochs = 50)


"""
works

data = np.arange(0,100,1).reshape(10,1,10)
labels = np.arange(1,101,1).reshape(10,1,10)

model.add(Embedding(100, 5, activation = 'selu', input_shape = (1, 10)))
model.add(RNN(10, activation="tanh", bptt_truncate = 3, input_shape = (10, 10)))

"""
