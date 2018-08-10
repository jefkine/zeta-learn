# -*- coding: utf-8 -*-

from sklearn import datasets

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.optimizers import register_opt
from ztlearn.dl.layers import Dropout, Dense, BatchNormalization

# NOTE: Check the random_seed seeding for improperly shuffled data.
data = datasets.load_digits()
train_data, test_data, train_label, test_label = train_test_split(data.data,
                                                                  data.target,
                                                                  test_size = 0.3, random_seed = 3)

# plot samples of training data
plot_tiled_img_samples(train_data, train_label)

# optimizer definition
opt = register_opt(optimizer_name = 'adam', momentum = 0.01, learning_rate = 0.001)

# model definition
model = Sequential()
model.add(Dense(256, activation = 'relu', input_shape=(64,)))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(10, activation = 'relu')) # 10 digits classes
model.compile(loss = 'cce', optimizer = opt)

model_epochs = 12
fit_stats = model.fit(train_data,
                      one_hot(train_label),
                      batch_size      = 128,
                      epochs          = model_epochs,
                      validation_data = (test_data, one_hot(test_label)),
                      shuffle_data    = True)

eval_stats  = model.evaluate(test_data, one_hot(test_label))
predictions = unhot(model.predict(test_data, True))
print_results(predictions, test_label)

plot_img_results(test_data, test_label, predictions)

model_name = 'digits_mlp'
plot_metric('loss',     model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'], model_name = model_name)
plot_metric('accuracy', model_epochs, fit_stats['train_acc'],  fit_stats['valid_acc'],  model_name = model_name)
plot_metric('evaluation',
                          eval_stats['valid_batches'],
                          eval_stats['valid_loss'],
                          eval_stats['valid_acc'],
                          model_name = model_name, legend = ['loss', 'acc'])
