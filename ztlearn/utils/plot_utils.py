# -*- coding: utf-8 -*-

import time
import matplotlib.pyplot as plt

SMALL_FONT = 10
LARGE_FONT = 14
FIG_SIZE   = (8, 6)

img_specs = {
    'mnist' :  {
        'pix_row'    : 1,
        'pix_col'    : 26,
        'img_width'  : 28,
        'img_height' : 28
    },
    'cifar' :  {
        'pix_row'    : 1,
        'pix_col'    : 30,
        'img_width'  : 32,
        'img_height' : 32
    },
    'digits':  {
        'pix_row'    : 0,
        'pix_col'    : 7,
        'img_width'  : 8,
        'img_height' : 8
    }
}

def plotter(x,
                y           = [],
                plot_dict   = {},
                fig_dims    = (7, 5),
                title       = 'Model',
                title_dict  = {},
                ylabel      = 'y-axis',
                ylabel_dict = {},
                xlabel      = 'x-axis',
                xlabel_dict = {},
                legend      = [], # ['train', 'valid'],
                legend_dict = {},
                file_path   = '',
                to_save     = False,
                plot_type   = 'line',
                cmap_name   = None,
                cmap_number = 10,
                grid_on     = True):

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_dims)

    ax.set_axisbelow(True)
    ax.minorticks_on()

    if grid_on:
        ax.grid(which = 'major', linestyle = '-', linewidth = 0.5, color = 'grey')
        ax.grid(which = 'minor', linestyle = ':', linewidth = 0.5, color = 'red')

    if plot_type == 'line':
        for i in range(len(y)):
            ax.plot(x, y[i], **plot_dict)

    if plot_type == 'scatter':
        if cmap_name is not None:
            plot_dict.update(cmap = plt.cm.get_cmap(cmap_name, cmap_number))
            plot = ax.scatter(x[:, 0], x[:, 1], **plot_dict)
            fig.colorbar(plot, ax = ax)
        else:
            ax.scatter(x[:, 0], x[:, 1], **plot_dict)
        if y is not None:
            ax.scatter(y[:, 0], y[:, 1], **{'c' : 'red'}) # centroids for k-means

    ax.set_title(title, **title_dict)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(legend, **legend_dict)

    if to_save:
        fig.savefig(file_path)

    return plt


def plot_pca(components,
                            n_components = 2,
                            colour_array = None,
                            model_name   = 'PCA',
                            to_save      = False,
                            fig_dims     = FIG_SIZE, #(10, 8),
                            title_dict   = {'size' : SMALL_FONT}):

    file_path = '../plots/decompositions/'+('{}{}{}{}{}'.format(model_name,
                                                               '_',
                                                               n_components,
                                                               '_Components_',
                                                               time.strftime("%Y-%m-%d_%H-%M-%S"),'.png'))

    plt_dict = {
        'c'         : colour_array,
        'edgecolor' : 'none',
        'alpha'     : 0.5,
        's'         : 50
    }

    plt = plotter(components,
                                y           = None,
                                plot_dict   = plt_dict,
                                fig_dims    = fig_dims,
                                title       = 'Model {}'.format(model_name.upper()),
                                title_dict  = title_dict,
                                xlabel      = 'PC 1',
                                ylabel      = 'PC 2',
                                file_path   = file_path,
                                to_save     = to_save,
                                plot_type   = 'scatter',
                                cmap_name   = 'tab10',
                                cmap_number = 10,
                                grid_on     = False)

    plt.show()


def plot_kmeans(data,
                        labels         = None,
                        centroids      = None,
                        model_name     = 'K-Means',
                        model_clusters = 1,
                        to_save        = False,
                        fig_dims       = FIG_SIZE,
                        title_dict     = {'size' : SMALL_FONT}):

    file_path = '../plots/clusters/'+('{}{}{}{}{}'.format(model_name,
                                                          '_',
                                                          model_clusters,
                                                          '_Clusters_',
                                                          time.strftime("%Y-%m-%d_%H-%M-%S"),'.png'))

    plt = plotter(data,
                        y          = centroids,
                        plot_dict  = {'c' : labels},
                        fig_dims   = fig_dims,
                        title      = 'Model {}'.format(model_name.title()),
                        title_dict = title_dict,
                        file_path  = file_path,
                        to_save    = to_save,
                        plot_type  = 'scatter')

    plt.show()


def plot_metric(metric,
                        epoch,
                        train,
                        valid,
                        model_name  = '',
                        to_save     = False,
                        plot_dict   = {'linewidth' : 0.8},
                        fig_dims    = FIG_SIZE,
                        title_dict  = {'size' : SMALL_FONT},
                        ylabel_dict = {'size' : SMALL_FONT},
                        xlabel_dict = {'size' : SMALL_FONT},
                        legend      = ['train', 'valid'],
                        legend_dict = {'loc' : 'upper right'}):

    file_path = '../plots/metrics/'+('{}{}{}{}{}'.format(model_name,
                                                         '_',
                                                         metric,
                                                         '_',
                                                         time.strftime("%Y-%m-%d_%H-%M-%S"),'.png'))

    plt = plotter(range(epoch),
                                [train, valid],
                                plot_dict   = plot_dict,
                                fig_dims    = fig_dims,
                                title       = 'Model {}'.format(metric.title()),
                                title_dict  = title_dict,
                                ylabel      = metric.title(),
                                ylabel_dict = ylabel_dict,
                                xlabel      = 'Iterations',
                                xlabel_dict = xlabel_dict,
                                legend      = legend,
                                legend_dict = legend_dict,
                                file_path   = file_path,
                                to_save     = to_save)

    plt.show()


def plot_opt_viz(dims,
                        x,
                        y,
                        z,
                        f_solution,
                        overlay     = 'plot',
                        to_save     = False,
                        title       = 'Optimization',
                        title_dict  = {'size' : LARGE_FONT},
                        fig_dims    = FIG_SIZE,
                        xticks_dict = {'size' : LARGE_FONT},
                        yticks_dict = {'size' : LARGE_FONT},
                        xlabel      = r'$\theta^1$',
                        xlabel_dict = {'size' : LARGE_FONT},
                        ylabel      = r'$\theta^2$',
                        ylabel_dict = {'size' : LARGE_FONT},
                        legend      = ['train', 'valid'],
                        legend_dict = {}):

    if dims == 3:
        fig = plt.figure(figsize = fig_dims)

        if overlay == 'wireframe':
            from mpl_toolkits.mplot3d import axes3d # for 3d projections
            ax = fig.add_subplot(111, projection = '3d')
            plt.scatter(y[:,0], y[:,1], s = f_solution, c = 'r')
            ax.plot_wireframe(x[0], x[1], z, rstride = 5, cstride = 5, linewidth = 0.5)

        elif overlay == 'contour':
            ax = fig.add_subplot(111)
            plt.scatter(y[:,0], y[:,1], s = f_solution, c = 'r')
            ax.contour(x[0], x[1], z, 20, cmap = plt.cm.jet)

        ax.set_xlabel(xlabel, **xlabel_dict)
        ax.set_ylabel(ylabel, **ylabel_dict)

    elif dims == 2:
        plt.figure(figsize = fig_dims)

        plt.xticks(**xticks_dict)
        plt.yticks(**yticks_dict)

        plt.plot(x, y)
        plt.scatter(z, f_solution, color = 'r')

        plt.xlabel(xlabel, **xlabel_dict)
        plt.ylabel(ylabel, **ylabel_dict)

    if to_save:
        plt.suptitle(('{}{}'.format(dims, 'D Surfaces')), fontsize = 14)
        plt.savefig('../plots/'+('{}{}{}{}'.format(overlay, '_', dims, 'd.png')))

    plt.show()


def plot_img_samples(train_data, train_target = None, fig_dims = (6, 6), dataset = 'digits', channels = 1):
    fig = plt.figure(figsize = fig_dims)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)


    for i in range(36):
        digit = fig.add_subplot(6, 6, i+1, xticks = [], yticks = [])

        if channels == 3:
            color_img = train_data[i].reshape(channels,
                                              img_specs[dataset]['img_height'],
                                              img_specs[dataset]['img_width']).transpose([1, 2, 0])
            digit.imshow(color_img, interpolation = 'nearest')
        else:
            digit.imshow(train_data[i].reshape(img_specs[dataset]['img_height'],
                                               img_specs[dataset]['img_width']),
                                               cmap = plt.cm.binary, interpolation = 'nearest')

        if train_target is not None:
            digit.text(img_specs[dataset]['pix_row'],
                       img_specs[dataset]['pix_col'],
                       str(train_target.astype('int')[i]))

    plt.show()


def plot_tiled_img_samples(train_data, train_target = None, fig_dims = (6, 6), dataset = 'digits', channels = 1):
    fig = plt.figure(figsize = fig_dims)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

    for i in range(36):
        digit = fig.add_subplot(6, 6, i+1)

        digit.grid(which = 'major', linestyle = ':', linewidth = 0.5, color = 'blue')
        digit.grid(which = 'minor', linestyle = ':', linewidth = 0.5, color = 'blue')

        digit.xaxis.set_ticklabels([])
        digit.yaxis.set_ticklabels([])

        digit.minorticks_on()

        if channels == 3:
            color_img = train_data[i].reshape(channels, 32, 32).transpose([1, 2, 0])
            digit.imshow(color_img, interpolation = 'nearest')
        else:
            digit.imshow(train_data[i].reshape(img_specs[dataset]['img_height'],
                                               img_specs[dataset]['img_width']),
                                               cmap = plt.cm.binary, interpolation = 'nearest')

        if train_target is not None:
            digit.text(img_specs[dataset]['pix_row'],
                       img_specs[dataset]['pix_col'],
                       str(train_target.astype('int')[i]))

    plt.show()


def plot_img_results(test_data, test_label, predictions, fig_dims = (6, 6), dataset = 'digits', channels = 1):
    fig = plt.figure(figsize = fig_dims)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

    for i in range(36):
        digit = fig.add_subplot(6, 6, i + 1, xticks = [], yticks = [])

        if channels == 3:
            color_img = test_data[i].reshape(channels,
                                             img_specs[dataset]['img_height'],
                                             img_specs[dataset]['img_width']).transpose([1, 2, 0])
            digit.imshow(color_img, interpolation = 'nearest')
        else:
            digit.imshow(test_data.reshape(-1,
                                           img_specs[dataset]['img_height'],
                                           img_specs[dataset]['img_width'])[i],
                                           cmap = plt.cm.binary, interpolation = 'nearest')

        if predictions[i] == test_label[i]:
            digit.text(img_specs[dataset]['pix_row'],
                       img_specs[dataset]['pix_col'],
                       str(predictions[i]), color = 'green')

        else:
            digit.text(img_specs[dataset]['pix_row'],
                       img_specs[dataset]['pix_col'],
                       str(predictions[i]), color = 'red')

    plt.show()


def plot_generated_img_samples(test_label,
                                            predictions,
                                            fig_dims   = (6, 6),
                                            dataset    = 'digits',
                                            channels   = 1,
                                            to_save    = False,
                                            iteration  = 0,
                                            model_name = ''):

    fig = plt.figure(figsize = fig_dims)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

    for i in range(36):
        digit = fig.add_subplot(6, 6, i+1, xticks = [], yticks = [])

        if channels == 3:
            color_img = predictions[i].reshape(channels,
                                               img_specs[dataset]['img_height'],
                                               img_specs[dataset]['img_width']).transpose([1, 2, 0])
            digit.imshow(color_img, interpolation = 'nearest')
        else:
            digit.imshow(predictions.reshape(-1,
                                             img_specs[dataset]['img_height'],
                                             img_specs[dataset]['img_width'])[i],
                                             cmap = plt.cm.binary, interpolation = 'nearest')

        if test_label is not None:
            digit.text(img_specs[dataset]['pix_row'],
                       img_specs[dataset]['pix_col'],
                       str(test_label[i]), color = 'blue')

    if to_save:
        plt.suptitle(('{}{}'.format('Generator Epoch: ', iteration)), y = 1.05, fontsize = 12).set_color('blue')
        plt.savefig('../plots/generated/'+('{}{}{}'.format(model_name, '_', iteration, '.png')))

    plt.show(block = False) if to_save else plt.show(block = True)


def plot_regression_results(train_data,
                                        train_label,
                                        test_data,
                                        test_label,
                                        input_data,
                                        pred_line,
                                        mse,
                                        super_title,
                                        y_label,
                                        x_label,
                                        model_name = '',
                                        to_save    = False,
                                        fig_dims   = FIG_SIZE,
                                        font_size  = 10):

    plt.figure(figsize = fig_dims)

    cmap  = plt.get_cmap('summer')
    train = plt.scatter(train_data, train_label, color = cmap(0.8), s = 12)
    test  = plt.scatter(test_data,  test_label,  color = cmap(0.4), s = 12)

    # minimum parameters: plt.plot(input_data, pred_line, '*', color = 'green', markersize = 4)
    plt.plot(input_data, pred_line, marker = '*', color = 'green', markersize = 4, linestyle = 'none')
    plt.suptitle(super_title)

    if mse is not None:
        plt.title("MSE: {:4.2f}".format(mse), size = font_size)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend((train, test), ("Train", "Test"), loc = 'upper left')

    if to_save:
        plt.savefig('../plots/metrics/'+('{}{}{:4.2f}{}{}{}'.format(model_name,
                                                                    '_mse_',
                                                                    mse,
                                                                    '_',
                                                                    time.strftime("%Y-%m-%d_%H-%M-%S"),'.png')))

    plt.show()
