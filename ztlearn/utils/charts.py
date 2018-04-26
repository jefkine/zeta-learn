# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

FONT_SIZE = 10
FIG_SIZE = (7, 5)

def plotter(x,
               y = [],
               plot_dict = {},
               fig_dims = (7, 5),
               xticks_dict = {},
               yticks_dict = {},
               title = 'Model',
               title_dict = {},
               ylabel = 'y',
               ylabel_dict = {},
               xlabel = 'x',
               xlabel_dict = {},
               legend = ['train', 'valid'],
               legend_dict = {}):

    plt.figure(figsize = fig_dims)
    for i in range(len(y)):
        plt.plot(x, y[i], **plot_dict)
    plt.xticks(**xticks_dict)
    plt.yticks(**yticks_dict)
    plt.title(title, **title_dict)
    plt.xlabel(xlabel, **xlabel_dict)
    plt.ylabel(ylabel, **ylabel_dict)
    plt.legend(legend, **legend_dict)

    return plt


def plot_metric(metric,
                        epoch,
                        train,
                        valid,
                        plot_dict = {'linewidth' : 0.8},
                        fig_dims = FIG_SIZE,
                        xticks_dict = {'size' : FONT_SIZE},
                        yticks_dict = {'size' : FONT_SIZE},
                        title_dict = {'size' : FONT_SIZE},
                        ylabel_dict = {'size' : FONT_SIZE},
                        xlabel_dict = {'size' : FONT_SIZE},
                        legend = ['train', 'valid'],
                        legend_dict = {'loc' : 'upper right'}):

    plt = plotter(range(epoch),
                                [train, valid],
                                plot_dict = plot_dict,
                                fig_dims = fig_dims,
                                xticks_dict = xticks_dict,
                                yticks_dict = yticks_dict,
                                title = 'Model {}'.format(metric.title()),
                                title_dict = title_dict,
                                ylabel = metric.title(),
                                ylabel_dict = ylabel_dict,
                                xlabel = 'Iterations',
                                xlabel_dict = xlabel_dict,
                                legend = legend,
                                legend_dict = legend_dict)

    plt.show()


def plot_digits_img_results(test_data, test_label, predictions, fig_dims = (6, 6)):
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


def plot_generated_digits_samples(test_label, predictions, fig_dims = (6, 6)):
    fig = plt.figure(figsize = fig_dims)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

    for i in range(36):
        digit = fig.add_subplot(6, 6, i+1, xticks = [], yticks = [])
        digit.imshow(predictions[i], cmap = plt.cm.binary, interpolation = 'nearest')
        if test_label is not None:
            digit.text(0, 7, str(test_label[i]), color = 'blue')

    plt.show()


def plot_digits_img_samples(data, fig_dims = (6, 6)):
    fig = plt.figure(figsize = fig_dims)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

    for i in range(36):
        digit = fig.add_subplot(6, 6, i+1, xticks = [], yticks = [])
        digit.imshow(data.images[i], cmap = plt.cm.binary, interpolation = 'nearest')
        digit.text(0, 7, str(data.target[i]))

    plt.show()


def plot_regression_results(train_data,
                                        train_label,
                                        test_data,
                                        test_label,
                                        input_data,
                                        pred_line,
                                        mse, super_title,
                                        y_label,
                                        x_label,
                                        fig_dims = FIG_SIZE,
                                        font_size = 10):

    plt.figure(figsize = fig_dims)
    cmap = plt.get_cmap('summer')
    train = plt.scatter(train_data, train_label, color = cmap(0.8), s = 12)
    test = plt.scatter(test_data, test_label, color = cmap(0.4), s = 12)
    plt.plot(input_data, pred_line, '*', color = 'green', markersize = 4)
    plt.suptitle(super_title)

    if mse is not None:
        plt.title("MSE: {:4.2f}".format(mse), size = font_size)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend((train, test), ("Train", "Test"), loc='upper left')

    plt.show()
