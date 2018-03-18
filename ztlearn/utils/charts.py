# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot_accuracy(epoch, train_acc, valid_acc, fig_dims = (7, 5), font_size = 10):
    plt.figure(figsize = fig_dims)
    plt.plot(range(epoch), train_acc, linewidth = 0.8)
    plt.plot(range(epoch), valid_acc, linewidth = 0.8)
    plt.xticks(size = font_size)
    plt.yticks(size = font_size)
    plt.title("Model Accuracy", size = font_size)
    plt.ylabel('Accuracy', size = font_size)
    plt.xlabel('Iterations', fontsize = font_size)
    plt.legend(['train', 'valid'], loc = 'upper right')
    plt.show()
    # plt.clf()

def plot_loss(epoch, train_loss, valid_loss, fig_dims = (7, 5), font_size = 10):
    plt.figure(figsize = fig_dims)
    plt.plot(range(epoch), train_loss, linewidth = 0.8)
    plt.plot(range(epoch), valid_loss, linewidth = 0.8)
    plt.xticks(size = font_size)
    plt.yticks(size = font_size)
    plt.title("Model Loss", size = font_size)
    plt.ylabel('Loss', size = font_size)
    plt.xlabel('Iterations', fontsize = font_size)
    plt.legend(['train', 'valid'], loc = 'upper right')
    plt.show()
    # plt.clf()

def plot_acc_loss(epoch, acc, loss, fig_dims = (7, 5), font_size = 10):
    plt.figure(figsize = fig_dims)
    plt.plot(range(epoch), acc, linewidth = 0.8)
    plt.plot(range(epoch), loss, linewidth = 0.8)
    plt.xticks(size = font_size)
    plt.yticks(size = font_size)
    plt.title("Model Loss Accuracy", size = font_size)
    plt.ylabel('Loss Accuracy', size = font_size)
    plt.xlabel('Iterations', fontsize = font_size)
    plt.legend(['loss', 'acc'], loc = 'upper right')
    plt.show()
    # plt.clf()

def plot_mnist_img_results(test_data, test_label, predictions, fig_dims = (6, 6)):
    fig = plt.figure(figsize = fig_dims)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

    for i in range(36):
        digit = fig.add_subplot(6, 6, i + 1, xticks = [], yticks = [])
        digit.imshow(test_data.reshape(-1, 8, 8)[i], cmap = plt.cm.binary, interpolation = 'nearest')

        if predictions[i] == test_label[i]:
            digit.text(0, 7, str(predictions[i]), color = 'green')
        else:
            digit.text(0, 7, str(predictions[i]), color = 'red')

    plt.show()
    # plt.clf()
    
def plot_generated_mnist_samples(data, fig_dims = (6, 6)):
    fig = plt.figure(figsize = fig_dims)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
    
    for i in range(36):
        digit = fig.add_subplot(6, 6, i+1, xticks = [], yticks = [])
        digit.imshow(data[i], cmap = plt.cm.binary, interpolation = 'nearest')
         
    plt.show()

def plot_mnist_img_samples(data, fig_dims = (6, 6)):
    fig = plt.figure(figsize = fig_dims)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

    for i in range(36):
        digit = fig.add_subplot(6, 6, i+1, xticks = [], yticks = [])
        digit.imshow(data.images[i], cmap = plt.cm.binary, interpolation = 'nearest')
        digit.text(0, 7, str(data.target[i]))

    plt.show()
    # plt.clf()

def plot_regression_results(train_data, train_label, test_data, test_label, input_data, pred_line, mse, super_title, y_label, x_label, fig_dims = (7, 5), font_size = 10):
    plt.figure(figsize = fig_dims)
    cmap = plt.get_cmap('summer')
    train = plt.scatter(train_data, train_label, color = cmap(0.8), s = 12)
    test = plt.scatter(test_data, test_label, color = cmap(0.4), s = 12)
    plt.plot(input_data, pred_line, '*', color = 'green', markersize = 4)
    plt.suptitle(super_title)
    
    if mse is not None:
        plt.title("MSE: {:4.2f}".format(mse), fontsize = font_size)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend((train, test), ("Train", "Test"), loc='upper left')
    plt.show()
    # plt.clf()

