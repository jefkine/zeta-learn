# -*- coding: utf-8 -*-

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.optimizers import register_opt
from ztlearn.datasets.cifar import fetch_cifar_100
from ztlearn.dl.layers import Dropout, Dense, BatchNormalization, Activation

data = fetch_cifar_100()
train_data, test_data, train_label, test_label = train_test_split(data.data,
                                                                  data.target,
                                                                  test_size   = 0.3,
                                                                  random_seed = 3)

# plot samples of training data
plot_img_samples(train_data, train_label, dataset = 'cifar', channels = 3)

reshaped_image_dims = 3 * 32 * 32 # ==> (channels * height * width)
reshaped_train_data = z_score(train_data.reshape(train_data.shape[0], reshaped_image_dims).astype('float32'))
reshaped_test_data  = z_score(test_data.reshape(test_data.shape[0], reshaped_image_dims).astype('float32'))

# optimizer definition
opt = register_opt(optimizer_name = 'adam', momentum = 0.01, lr = 0.0001)

model = Sequential()
model.add(Dense(1024, input_shape = (3072, )))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(100))
model.add(Activation('softmax'))
model.compile(loss = 'cce', optimizer = opt)

model.summary(model_name = 'cifar-100 mlp')

model_epochs = 12 # change to 200 epochs
fit_stats = model.fit(reshaped_train_data,
                      one_hot(train_label),
                      batch_size      = 128,
                      epochs          = model_epochs,
                      validation_data = (reshaped_test_data, one_hot(test_label)),
                      shuffle_data    = True)

eval_stats  = model.evaluate(reshaped_test_data, one_hot(test_label))
predictions = unhot(model.predict(reshaped_test_data, True))
print_results(predictions, test_label)

plot_img_results(test_data, test_label, predictions, dataset = 'cifar', channels = 3)

model_name = model.model_name
plot_metric('loss', model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'], model_name = model_name)
plot_metric('accuracy', model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'], model_name = model_name)
plot_metric('evaluation',
                          eval_stats['valid_batches'],
                          eval_stats['valid_loss'],
                          eval_stats['valid_acc'],
                          model_name = model_name,
                          legend     = ['loss', 'acc'])
