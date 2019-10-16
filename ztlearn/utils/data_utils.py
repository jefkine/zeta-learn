# -*- coding: utf-8 -*-

import os
import sys
import gzip
import urllib
import tarfile
import zipfile
import numpy as np

from itertools import chain
from itertools import combinations
from itertools import combinations_with_replacement

#-----------------------------------------------------------------------------#
#                       DATA UTILITY FUNCTIONS                                #
#-----------------------------------------------------------------------------#

def eucledian_norm(vec_a, vec_b):
    """ compute the eucledian distance between two vectors """
    distance = vec_a - vec_b
    # return np.sqrt(np.sum(np.square(distance)))
    # return np.sqrt(np.einsum('ij, ij->i', distance, distance))
    return np.linalg.norm(distance, ord = 'fro', axis = 1)

def clip_gradients(grad, g_min = -1., g_max = 1.):
    """ enforce min and max bounderies on a given gradient """
    return np.clip(grad, g_min, g_max, out = grad)

def accuracy_score(predictions, targets):
    """ compute an average accuracy score of prediction vs targets """
    return np.mean(predictions == targets)

def one_hot(labels, num_classes = None):
    """ generate one hot encoding for a given set labels """
    num_classes    = np.max(labels.astype('int')) + 1 if not num_classes else num_classes
    one_hot_labels = np.zeros([labels.size, num_classes])

    one_hot_labels[np.arange(labels.size), labels.astype('int')] = 1.

    return one_hot_labels

def unhot(one_hot, unhot_axis = 1):
    """ reverse one hot encoded data back to labels """
    return np.argmax(one_hot, axis = unhot_axis)

def shuffle_data(input_data, input_label, random_seed = None):
    """ perfom randomized shuffle on a given input dataset """
    assert input_data.shape[0] == input_label.shape[0], 'input data and label sizes do not match!'

    if random_seed is not None: # confirm that random seed has been set
        np.random.seed(random_seed)

    indices = np.arange(input_data.shape[0])
    np.random.shuffle(indices)

    return input_data[indices], input_label[indices]

def train_test_split(samples, labels, test_size = 0.2, shuffle = True, random_seed = None, cut_off = None):
    """ generate a train vs test split given a test size """
    if shuffle:
        samples, labels = shuffle_data(samples, labels, random_seed)

    split_ratio = int((1.0 - test_size) * len(samples))

    samples_train, samples_test = samples[:split_ratio], samples[split_ratio:]
    labels_train, labels_test   = labels[:split_ratio], labels[split_ratio:]

    if cut_off is not None and isinstance(cut_off, (int, np.integer)):
        return samples_train[:cut_off], samples_test[:cut_off], labels_train[:cut_off], labels_test[:cut_off]

    return samples_train, samples_test, labels_train, labels_test

def kfold_split(samples, labels, n_splits = 5, shuffle = False, random_seed = None):
    """ generate K folds for cross validation testing """
    if shuffle:
        samples, labels = shuffle_data(samples, labels, random_seed)

    def get_folds(input_data, num_rows):
        fold_size = int(num_rows / n_splits)
        for idx in range(0, num_rows, fold_size):
            yield input_data[idx:idx + fold_size]

    sample_folds = list(get_folds(samples, samples.shape[0]))
    label_folds  = list(get_folds(labels, samples.shape[0]))

    # @@TODO: split the sample_folds and label_folds into sizes K-1 and 1 for test and training sets
    # this returns two lists which can be accessed by the item index e.g sample_folds[0] for fold 1
    return sample_folds, label_folds


def minibatches(input_data, input_label, batch_size, shuffle):
    """ generate minibatches on a given input data matrix """
    assert input_data.shape[0] == input_label.shape[0], 'input data and label sizes do not match!'
    minibatches = []
    indices     = np.arange(input_data.shape[0])

    if shuffle:
        np.random.shuffle(indices)

    for idx in range(0, input_data.shape[0], batch_size):
        mini_batch = indices[idx:idx + batch_size]
        minibatches.append((input_data[mini_batch], input_label[mini_batch]))

    return minibatches

def normalize(input_data, axis = -1, order = 2):
    """ compute normalization (order) for a given input matrix, order and axis """
    l2 = np.linalg.norm(input_data, order, axis, keepdims = True)
    l2[l2 == 0] = 1

    return input_data / l2

def range_normalize(input_data, a = -1, b = 1, axis = None):
    """ compute the range normalization for a given input matrix, range [a,b] and axis """
    return (((b - a) * ((input_data - input_data.min(axis = axis, keepdims = True)) / np.ptp(input_data, axis = axis))) + a)

def min_max(input_data, axis = None):
    """ compute the min max standardization for a given input matrix and axis """
    return (input_data - input_data.min(axis = axis, keepdims = True)) / np.ptp(input_data, axis = axis)

def z_score(input_data, axis = None):
    """ compute the z score for a given input matrix and axis """
    input_mean = input_data.mean(axis = axis, keepdims = True)
    input_std  = input_data.std(axis = axis, keepdims = True)

    return (input_data - input_mean) / input_std

def print_results(predictions, test_labels, num_samples = 20):
    """ print model targeted vs predicted results """
    print('Targeted  : {}'.format(test_labels[:num_samples]))
    print('Predicted : {}{}'.format(predictions[:num_samples], print_pad(1)))
    print('Model Accuracy : {:2.2f}% {}'.format(accuracy_score(predictions, test_labels)*100, print_pad(1)))

def print_seq_samples(train_data, train_label, unhot_axis = 1, sample_num = 0):
    """ print generated sequence samples """
    print(print_pad(1) + 'Sample Sequence : {}'.format(unhot(train_data[sample_num])))
    print('Next Entry      : {} {}'.format(unhot(train_label[sample_num], unhot_axis), print_pad(1)))

def print_seq_results(predicted, test_label, test_data, unhot_axis = 1, interval = 5):
    """ print results for a model predicting a sequence """
    predictions = unhot(predicted, unhot_axis)
    targets     = unhot(test_label, unhot_axis)

    for i in range(interval):
        print('Sequence  : {}'.format(unhot(test_data[i])))
        print('Targeted  : {}'.format(targets[i]))
        print('Predicted : {} {}'.format(predictions[i], print_pad(1)))

    print('Model Accuracy : {:2.2f}%'.format(accuracy_score(predictions, targets)*100))

def computebar(total, curr, size = 45, sign = "#", prefix = "Computing"):
    """ generate a graphical loading bar [####---] for a given iteration """
    progress = float((curr + 1) / total)
    update   = int(round(size * progress))

    bar = "\r{}: [{}] {:d}% {}".format(prefix,
                                       sign * update + "-" * (size - update),
                                       int(round(progress * 100)),
                                       "" if progress < 1. else print_pad(1, "\r\n"))

    sys.stdout.write(bar)
    sys.stdout.flush()

def print_pad(pad_count, pad_char = "\n"):
    """ pad strings with a total of n = pad_count, pad_char type characters """
    padding = ""
    for i in range(pad_count):
        padding += pad_char
    return padding

def custom_tuple(tup):
    """ customize tuple to have comma separated numbers """
    tuple_string = "("
    for itup in tup:
        tuple_string += "{:,d}".format(itup) + ", "

    if len(tup) == 1:
        return tuple_string[:-2] + ",)"
    return tuple_string[:-2] + ")"

def maybe_download(path, url, print_log = False):
    """ download the data from url, or return existing """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, url.rpartition('/')[2])
    if os.path.exists(filepath):
        if print_log:
            print('{} already exists'.format(filepath))
    else:
        print(print_pad(1) +'Required Dataset Not Found! Will Proceed To Download.')
        print(print_pad(1) +'Downloading. Please Wait ...')
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        print(print_pad(1) +'Succesful Download. See : {}'.format(filepath))

    return filepath

def extract_files(path, filepath):
    """ extract files from a detected compressed format """
    opener, mode = None, None
    if filepath.endswith('.tar.bz2') or filepath.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    elif zipfile.is_zipfile(filepath):
        opener, mode = zipfile.ZipFile, 'r'
    elif filepath.endswith('.tar.gz') :
        opener, mode = tarfile.open, 'r:gz'
    elif filepath.endswith('.gz'): # @@TODO: write a more robust method for this
        opener, mode = gzip.open, 'rb'

    if opener is not None:
        if opener is gzip.open: # OR if mode == 'rb':
            with opener(filepath, mode) as f:
                contents = f.read()
            with open(os.path.splitext(filepath)[0], 'wb') as f:
                f.write(contents)
        else:
            with opener(filepath, mode) as f:
                f.extractall(path)
    else:
        raise NotImplementedError('Extraction Method For This Filetype Not Implemented')

def polynomial_features(inputs, degree = 2, repeated_elems = False, with_bias = True):
    """ generate feature matrix of all polynomial combinations for degrees upto <= degree """
    num_samples, num_features = np.shape(inputs)

    def feature_combinations():
        combination_type = (combinations if repeated_elems else combinations_with_replacement)
        start = 0 if repeated_elems == False else 1
        combs = [list(combination_type(range(num_features), i)) for i in range(start, degree + 1)]
        flatten_combs =  list(chain(*combs))
        return flatten_combs, len(flatten_combs)

    flat_combs, num_output_features = feature_combinations()
    inputs_poly = np.empty((num_samples, num_output_features), dtype = inputs.dtype)
    for i, flat_comb in enumerate(flat_combs):
        inputs_poly[:, i] = inputs[:, flat_comb].prod(1)

    return inputs_poly
