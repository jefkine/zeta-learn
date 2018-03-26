# -*- coding: utf-8 -*-

import sys
import numpy as np

#-----------------------------------------------------------------------------#
#                       DATA UTILITY FUNCTIONS                                #
#-----------------------------------------------------------------------------#

def clip_gradients(grad, g_min = -1., g_max = 1.):
    return np.clip(grad, g_min, g_max, out = grad)

def accuracy_score(predictions, targets):
    return np.mean(predictions == targets)

def one_hot(labels, num_classes = None):
    num_classes = np.max(labels) + 1 if not num_classes else num_classes
    one_hot_labels = np.zeros([labels.size, num_classes])
    one_hot_labels[np.arange(labels.size), labels] = 1.
    return one_hot_labels

def unhot(one_hot, unhot_axis = 1):
    return np.argmax(one_hot, axis = unhot_axis)

def shuffle_data(input_data, input_label, random_seed = None):
    assert input_data.shape[0] == input_label.shape[0], 'input data and label sizes do not match!'
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(input_data.shape[0])
    np.random.shuffle(indices)
    return input_data[indices], input_label[indices]

def train_test_split(samples, labels, test_size = 0.2, shuffle = True, random_seed = None):
    if shuffle:
        samples, labels = shuffle_data(samples, labels, random_seed)
    split_ratio = int((1.0 - test_size) * len(samples))
    samples_train, samples_test = samples[:split_ratio], samples[split_ratio:]
    labels_train, labels_test = labels[:split_ratio], labels[split_ratio:]
    return samples_train, samples_test, labels_train, labels_test

def minibatches(input_data, input_label, batch_size, shuffle):
    assert input_data.shape[0] == input_label.shape[0], 'input data and label sizes do not match!'
    minibatches = []
    indices = np.arange(input_data.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    for idx in range(0, input_data.shape[0], batch_size):
        mini_batch = indices[idx:idx + batch_size]
        minibatches.append((input_data[mini_batch], input_label[mini_batch]))
    return minibatches

def normalize(input_data, axis = -1, order = 2):
    l2 = np.linalg.norm(input_data, order, axis, keepdims = True)
    l2[l2 == 0] = 1
    return input_data / l2

def min_max(input_data, axis = None):
    min_input = input_data.min(axis = axis, keepdims = True)
    max_input = input_data.max(axis = axis, keepdims = True)
    return (input_data - min_input)/(max_input - min_input)

def z_score(input_data, axis = None):
    input_mean = input_data.mean(axis = axis, keepdims = True)
    input_std = input_data.std(axis = axis, keepdims = True)
    return (input_data - input_mean) / input_std

def print_results(predictions, test_labels, num_samples = 20):
    print('Targeted  : {}'.format(test_labels[:num_samples]))
    print('Predicted : {}'.format(predictions[:num_samples]))
    print ('Model Accuracy : {:2.2f}% \n'.format(accuracy_score(predictions, test_labels)*100))
    
def print_seq_samples(train_data, train_label, unhot_axis = 1, sample_num = 0):
    print('Sample Sequence : {}'.format(unhot(train_data[sample_num])))
    print('Next Entry      : {} \n'.format(unhot(train_label[sample_num], unhot_axis)))
    
def print_seq_results(predicted, test_label, test_data, unhot_axis = 1, interval = 5):
    predictions = unhot(predicted, unhot_axis)
    targets = unhot(test_label, unhot_axis)
    
    for i in range(interval):
        print('Sequence  : {}'.format(unhot(test_data[i])))
        print('Targeted  : {}'.format(targets[i]))
        print('Predicted : {} \n'.format(predictions[i]))
        
    print ('Model Accuracy : {:2.2f}%'.format(accuracy_score(predictions, targets)*100))
     
def computebar(total, curr, size = 40, sign = "#", prefix = "Computing"):
    progress = float((curr + 1) / total)        
    update = int(round(size * progress))  
    
    bar = "\r{}: [{}] {:.0f}% {}".format(prefix, 
                                         sign * update + "-" * (size - update), 
                                         round(progress * 100, 0), 
                                         "" if progress < 1. else '\r\n')
   
    sys.stdout.write(bar)
    sys.stdout.flush()
