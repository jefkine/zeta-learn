# -*- coding: utf-8 -*-

from sklearn import datasets

from zeta.utils import unhot
from zeta.utils import one_hot
from zeta.utils import normalize
from zeta.utils import plot_acc_loss
from zeta.utils import print_results
from zeta.utils import train_test_split
from zeta.utils import plot_mnist_img_results
from zeta.utils import plot_mnist_img_samples

from zeta.dl.optimizers import register_opt
from zeta.ml.classification import Perceptron

data = datasets.load_digits()
plot_mnist_img_samples(data)

train_data, test_data, train_label, test_label = train_test_split(normalize(data.data),
                                                                  one_hot(data.target),
                                                                  test_size = 0.3,
                                                                  random_seed = 5)

opt = register_opt(optimizer_name = 'sgd-momentum', momentum = 0.01, learning_rate = 0.001)
model = Perceptron(epochs = 300, activation = 'selu',
                                 loss = 'categorical-cross-entropy',
                                 init_method = 'he-normal',
                                 optimizer = opt)

fit_stats = model.fit(train_data, train_label, verbose = False)

predictions = unhot(model.predict(test_data))
print_results(predictions, unhot(test_label))
plot_mnist_img_results(test_data, unhot(test_label), predictions)

plot_acc_loss(len(fit_stats["train_loss"]), fit_stats['train_acc'], fit_stats['train_loss'])

