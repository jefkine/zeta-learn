# -*- coding: utf-8 -*-

from sklearn import datasets

from ztlearn.utils import *
from ztlearn.dl.layers import Dense
from ztlearn.dl.models import Sequential
from ztlearn.optimizers import register_opt


data = datasets.load_iris()
# print(data['DESCR'])

train_data, test_data, train_label, test_label = train_test_split(data.data, data.target,
                                                                             test_size = 0.3, random_seed = 5)

opt = register_opt(optimizer_name = 'adam', momentum = 0.1, learning_rate = 0.01)

model = Sequential()
model.add(Dense(10, activation = 'sigmoid', input_shape = (train_data.shape[1],)))
model.add(Dense(3, activation = 'sigmoid')) # 3 iris_classes
model.compile(loss = 'categorical_crossentropy', optimizer = opt)

model_epochs = 25
fit_stats = model.fit(train_data,
                      one_hot(train_label),
                      batch_size      = 10,
                      epochs          = model_epochs,
                      validation_data = (test_data, one_hot(test_label)),
                      shuffle_data    = True)

# eval_stats = model.evaluate(test_data, one_hot(train_label))
predictions = unhot(model.predict(test_data))
print_results(predictions, test_label)

model_name = 'iris_mlp'
plot_metric('loss', model_epochs, fit_stats['train_loss'], fit_stats['valid_loss'], model_name = model_name)
plot_metric('accuracy', model_epochs, fit_stats['train_acc'], fit_stats['valid_acc'], model_name = model_name)
