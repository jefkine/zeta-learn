# -*- coding: utf-8 -*-

from ztlearn.utils import *
from ztlearn.optimizers import register_opt
from ztlearn.ml.classification import Perceptron
from ztlearn.datasets.cifar import fetch_cifar_100

data = fetch_cifar_100()
train_data, test_data, train_label, test_label = train_test_split(data.data,
                                                                  one_hot(data.target),
                                                                  test_size   = 0.3,
                                                                  random_seed = 5)

# plot samples of training data
plot_img_samples(train_data, unhot(train_label), dataset = 'cifar', channels = 3)

reshaped_image_dims = 3 * 1024 # ==> (channels * (height * width))
reshaped_train_data = z_score(train_data.reshape(train_data.shape[0], reshaped_image_dims).astype(np.float32))
reshaped_test_data  = z_score(test_data.reshape(test_data.shape[0], reshaped_image_dims).astype(np.float32))

# optimizer definition
opt = register_opt(optimizer_name = 'adam', momentum = 0.001, learning_rate = 0.0001)

# model definition
model     = Perceptron(epochs = 300, activation = 'relu', loss = 'cce', init_method = 'he_normal', optimizer = opt)
fit_stats = model.fit(reshaped_train_data, train_label)

predictions = unhot(model.predict(reshaped_test_data))
print_results(predictions, unhot(test_label))

plot_img_results(test_data, unhot(test_label), predictions, dataset = 'cifar', channels = 3)
plot_metric('accuracy_loss',
                              len(fit_stats["train_loss"]),
                              fit_stats['train_acc'],
                              fit_stats['train_loss'],
                              model_name = 'cifa_100_perceptron',
                              legend     = ['acc', 'loss'])
