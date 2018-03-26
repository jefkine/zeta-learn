# -*- coding: utf-8 -*-

from sklearn import datasets

from ztlearn.utils import *
from ztlearn.dl.optimizers import register_opt
from ztlearn.ml.classification import Perceptron

data = datasets.load_digits()
plot_digits_img_samples(data)

train_data, test_data, train_label, test_label = train_test_split(normalize(data.data),
                                                                  one_hot(data.target),
                                                                  test_size = 0.3,
                                                                  random_seed = 5)

opt = register_opt(optimizer_name = 'sgd_momentum', momentum = 0.01, learning_rate = 0.001)
model = Perceptron(epochs = 300, activation = 'selu',
                                 loss = 'categorical_crossentropy',
                                 init_method = 'he_normal',
                                 optimizer = opt)

fit_stats = model.fit(train_data, train_label)

predictions = unhot(model.predict(test_data))
print_results(predictions, unhot(test_label))
plot_digits_img_results(test_data, unhot(test_label), predictions)
plot_acc_loss(len(fit_stats["train_loss"]), fit_stats['train_acc'], fit_stats['train_loss'])
