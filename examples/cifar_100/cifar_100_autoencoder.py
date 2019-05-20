# -*- coding: utf-8 -*-

import numpy as np

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.optimizers import register_opt
from ztlearn.datasets.cifar import fetch_cifar_100
from ztlearn.dl.layers import BatchNormalization, Dense

img_rows   = 32
img_cols   = 32
img_dim    = 3072 # channels * img_rows * img_cols
channels   = 3    # Red Blue Green
latent_dim = 4
init_type  = 'he_uniform'

def stack_encoder_layers(init):
    model = Sequential(init_method = init)
    model.add(Dense(512, activation = 'relu', input_shape = (img_dim,)))
    model.add(BatchNormalization())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(latent_dim, activation = 'relu'))

    return model

def stack_decoder_layers(init):
    model = Sequential(init_method = init)
    model.add(Dense(256, activation = 'relu', input_shape = (latent_dim,)))
    model.add(BatchNormalization())
    model.add(Dense(512, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(img_dim, activation = 'sigmoid'))

    return model

encoder = stack_encoder_layers(init = init_type)
decoder = stack_decoder_layers(init = init_type)

opt = register_opt(optimizer_name = 'adam', momentum = 0.01, lr = 0.001)

autoencoder = Sequential(init_method = init_type)
autoencoder.layers.extend(encoder.layers)
autoencoder.layers.extend(decoder.layers)
autoencoder.compile(loss = 'categorical_crossentropy', optimizer = opt)

encoder.summary('cifar-100 encoder')
decoder.summary('cifar-100 decoder')

autoencoder.summary('cifar-100 autoencoder')

data   = fetch_cifar_100()
train_data, test_data, train_label, test_label = train_test_split(data.data,
                                                                  data.data,
                                                                  test_size   = 0.2,
                                                                  random_seed = 5,
                                                                  cut_off     = 2000)

# plot samples of training data
plot_img_samples(train_data, None, dataset = 'cifar', channels = 3)

transformed_image_dims  = img_dim
transformed_train_data  = z_score(train_data.reshape(train_data.shape[0], transformed_image_dims).astype(np.float32))
transformed_train_label = z_score(train_label.reshape(train_label.shape[0], transformed_image_dims).astype(np.float32))
transformed_test_data   = z_score(test_data.reshape(test_data.shape[0], transformed_image_dims).astype(np.float32))
transformed_test_label  = z_score(test_label.reshape(test_label.shape[0], transformed_image_dims).astype(np.float32))

model_epochs = 500
fit_stats = autoencoder.fit(transformed_train_data,
                            transformed_train_label,
                            batch_size      = 128,
                            epochs          = model_epochs,
                            validation_data = (transformed_test_data, transformed_test_label),
                            shuffle_data    = True)

# generate non rescaled test labels for use in generated digits plot (use the same random_seed as above)
_, _, _, test_label = train_test_split(data.data, data.target, test_size = 0.2, random_seed = 5)
predictions         = autoencoder.predict(transformed_test_data).reshape((-1, channels, img_rows, img_cols))

plot_generated_img_samples(unhot(one_hot(test_label)),
                                                        predictions,
                                                        dataset    = 'cifar',
                                                        channels   = 3,
                                                        to_save    = False,
                                                        iteration  = model_epochs,
                                                        model_name = autoencoder.model_name)

plot_metric('loss', model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'], model_name = autoencoder.model_name)
plot_metric('accuracy', model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'], model_name = autoencoder.model_name)
