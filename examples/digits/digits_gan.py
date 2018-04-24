# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets

from ztlearn.utils import *
from ztlearn.dl.models import Sequential
from ztlearn.dl.optimizers import register_opt
from ztlearn.dl.layers import BatchNormalization, Dense, Dropout, Activation


data = datasets.load_digits()
# plot_digits_img_samples(data)

img_rows = 8
img_cols = 8
img_dim = 64  # is the product (img_rows * img_cols)

latent_dim = 100
batch_size = 128
half_batch = int(batch_size * 0.5)

verbose = True
init_type = 'he_uniform'

model_epochs = 7500
model_stats = {'d_train_loss': [], 'd_train_acc': [], 'g_train_loss': [], 'g_train_acc': []}

d_opt = register_opt(optimizer_name = 'adam', beta1 = 0.5, learning_rate = 0.001)
g_opt = register_opt(optimizer_name = 'adam', beta1 = 0.5, learning_rate = 0.0001)

def stack_generator_layers(init):
    model = Sequential(init_method = init)
    model.add(Dense(128, input_shape = (latent_dim,)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum = 0.8))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum = 0.8))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum = 0.8))
    model.add(Dense(img_dim, activation = 'tanh'))

    return model

def stack_discriminator_layers(init):
    model = Sequential(init_method = init)
    model.add(Dense(256, input_shape = (img_dim,)))
    model.add(Activation('leaky_relu', alpha = 0.2))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Activation('leaky_relu', alpha = 0.2))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation = 'sigmoid'))

    return model

# stack and compile the generator
generator = stack_generator_layers(init = init_type)
generator.compile(loss = 'cce', optimizer = g_opt)

# stack and compile the discriminator
discriminator = stack_discriminator_layers(init = init_type)
discriminator.compile(loss = 'cce', optimizer = d_opt)

# stack and compile the generator_discriminator
generator_discriminator = Sequential(init_method = init_type)
generator_discriminator.layers.extend(generator.layers)
generator_discriminator.layers.extend(discriminator.layers)
generator_discriminator.compile(loss = 'cce', optimizer = g_opt)

# rescale to range [-1, 1]
images = range_normalize(data.data.astype(np.float32))

for epoch_idx in range(model_epochs):

    # set the discriminator to trainable
    discriminator.trainable = True

    for epoch_k in range(10):

        # draw random samples from real images
        index = np.random.choice(images.shape[0], half_batch, replace = False)
        # index = np.random.randint(0, images.shape[0], half_batch)
        imgs = images[index]

        d_noise = np.random.normal(0, 1, (half_batch, latent_dim))

        # generate a batch of new images
        gen_imgs = generator.predict(d_noise)

        # valid = [1, 0], fake = [0, 1]
        d_valid = np.concatenate((np.ones((half_batch, 1)), np.zeros((half_batch, 1))), axis = 1)
        d_fake = np.concatenate((np.zeros((half_batch, 1)), np.ones((half_batch, 1))), axis = 1)

        # discriminator training
        d_loss_real, d_acc_real = discriminator.train_on_batch(imgs, d_valid)
        d_loss_fake, d_acc_fake = discriminator.train_on_batch(gen_imgs, d_fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)

        if verbose:
            print('Epoch {} K:{} Discriminator Loss: {:2.4f}, Acc: {:2.4f}.'.format(epoch_idx+1, epoch_k+1, d_loss, d_acc))

    # end of for epoch_k in range(1):

    model_stats['d_train_loss'].append(d_loss)
    model_stats['d_train_acc'].append(d_acc)

    # set the discriminator to not trainable
    discriminator.trainable = False

    # discriminator training
    g_noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # g_valid = [1, 0]
    g_valid = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))), axis = 1)

    # train the generator
    g_loss, g_acc = generator_discriminator.train_on_batch(g_noise, g_valid)

    model_stats['g_train_loss'].append(g_loss)
    model_stats['g_train_acc'].append(g_acc)

    if not verbose:
        computebar(model_epochs, epoch_idx)
    else:
        # print the progress
        print_epoch = epoch_idx + 1
        print('\nEpoch {} Discriminator Loss: {:2.4f}, Acc: {:2.4f}.'.format(print_epoch, d_loss, d_acc))
        print('Epoch {} Generator Loss: {:2.4f}, Acc: {:2.4f}.\n'.format(print_epoch, g_loss, g_acc))

plot_metric('Loss', model_epochs, model_stats['d_train_loss'], model_stats['g_train_loss'], legend = ['D', 'G'])
plot_metric('Accuracy', model_epochs, model_stats['d_train_acc'], model_stats['g_train_acc'], legend = ['D', 'G'])

noise = np.random.normal(0, 1, (36, latent_dim))
gen_imgs = generator.predict(noise).reshape((-1, img_rows, img_cols))
plot_generated_digits_samples(None, gen_imgs)
