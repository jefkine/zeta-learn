import numpy as np

from ztlearn.utils import plot_opt_viz
from ztlearn.dl.initializers import InitializeWeights as init
from ztlearn.dl.optimizers import OptimizationFunction as optimize

class GbOptimization(object):

    def __init__(self, optimizer, init_method = 'ones'):
        self.optimizer = optimizer
        self.init_method = init_method

    def run(self, f, df, params = 1, epochs = 10, tol = 1e-4, scale_factor = 3, verbose = False):
        self.inputs = init(self.init_method).initialize_weights((params, 1)) * scale_factor
        self.f0 = f(self.inputs) # initial function value (fsolve)
        self.epochs = epochs

        self.fsolve = np.zeros((self.epochs, 1))
        self.weights = np.zeros((self.epochs, 1, params))

        for i in np.arange(self.epochs):
            self.inputs = optimize(self.optimizer).update(self.inputs, df(self.inputs))
            self.weights[i,:,:] = self.inputs.T

            f_solution = f(self.inputs)
            self.fsolve[i,:] = f_solution

            if verbose:
                if i%5 == 0:
                    print('Epoch-{} weights: {:.20}'.format(i+1, self.npstring(self.inputs.T)))
                    print('Epoch-{} eps: {:.20}'.format(i+1, self.npstring(self.f0 - f_solution)))
                # if np.linalg.norm(self.inputs, axis = 0) > tol: break

    def npstring(self, np_array):
        return np.array2string(np_array, formatter = {'float_kind':'{0:.4f}'.format})

    def plot_3d(self, f):
        """ plot a 3d visualization """
        theta = np.arange(-4.0, 4.0, 0.1)

        x_grid = np.meshgrid(theta, theta)
        z = f(x_grid)

        weights = self.weights.reshape(self.epochs, -1)

        vis_type = ['wireframe', 'contour']
        for vis in vis_type:
            plot_opt_viz(3,
                            x_grid,
                            weights,
                            z,
                            self.fsolve,
                            overlay = vis)

    def plot_2d(self, f):
        """ plot a 2d visualization """
        theta = np.expand_dims(np.arange(-5.0, 6.0, 1.0), axis = 1)

        y = np.zeros_like(theta)
        for i in np.arange(theta.shape[0]):
            y[i,:] = f(theta[i,:])

        weights = self.weights.reshape(self.epochs, -1)

        plot_opt_viz(2,
                         theta,
                         y,
                         weights,
                         self.fsolve,
                         overlay = 'plot')
