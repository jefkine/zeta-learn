import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pylab
from mpl_toolkits.mplot3d import Axes3D

from ztlearn.dl.initializers import InitializeWeights as init
from ztlearn.dl.optimizers import OptimizationFunction as optimize

class GbOptimization(object):

    def __init__(self, optimizer, init_method = 'ones'):
        self.optimizer = optimizer
        self.init_method = init_method

    def run(self, f, df, params = 1, epochs = 10, tol = 1e-4, verbose = False):
        self.inputs = init(self.init_method).initialize_weights((params, 1))
        self.inputs *= 3
        self.f0 = f(self.inputs) # initial function value (fsolve)
        self.df = df
        self.epochs = epochs
        self.tol = tol
        self.fsolve = np.zeros((self.epochs, 1))
        self.weights = np.zeros((self.epochs, 1, params))

        for i in np.arange(self.epochs):
            self.inputs = optimize(self.optimizer).update(self.inputs, self.df(self.inputs))
            self.weights[i,:,:] = self.inputs.T

            f_solution = f(self.inputs)
            self.fsolve[i,:] = f_solution
            eps = self.f0 - f_solution

            if verbose:
                if i%5 == 0:
                    print('Epoch-{} weights: {:.20}'.format(i+1, self.npstring(self.inputs.T)))
                    print('Epoch-{} eps: {:.20}'.format(i+1, self.npstring(eps)))
    #            if np.linalg.norm(self.inputs, axis = 0) > self.tol: break

    def npstring(self, np_array):
        return np.array2string(np_array, formatter = {'float_kind':'{0:.4f}'.format})

    def plot_3d(self, f):
        theta = np.arange(-4.0, 4.0, 0.1)
        x_grid = np.meshgrid(theta, theta)
        z = f(x_grid)
        weights = self.weights.reshape(self.epochs, -1)

        pylab.clf()
        fig = pylab.figure(figsize = (5, 7))

        ax1 = fig.add_subplot(211, projection = '3d')
        ax1.scatter(weights[:,0], weights[:,1], self.fsolve, color = 'r')
        ax1.plot_wireframe(x_grid[0], x_grid[1], z, rstride = 5, cstride = 5, linewidth = 0.5)
        ax1.set_xlabel(r'$\theta^1$',fontsize = 14)
        ax1.set_ylabel(r'$\theta^2$',fontsize = 14)

        ax2 = fig.add_subplot(212)
        ax2.scatter(weights[:,0], weights[:,1], self.fsolve, color = 'r')
        ax2.contour(x_grid[0], x_grid[1], z, 20, cmap = plt.cm.jet)

        pylab.suptitle('3D Surface',fontsize = 14)
        pylab.savefig('../plots/3d.png')
        pylab.show()

    def plot_2d(self, f):
        theta = np.arange(-5.0, 6.0, 1.0)
        theta = np.expand_dims(theta, axis = 1)
        y = np.zeros_like(theta)
        for i in np.arange(theta.shape[0]):
            y[i,:] = f(theta[i,:])

        weights = self.weights.reshape(self.epochs, -1)

        pylab.clf()
        pylab.plot(theta, y)
        weights = self.weights.reshape(self.epochs, -1)
        pylab.scatter(weights, self.fsolve, color = 'r')

        pylab.xlabel(r'$\theta$',fontsize = 14)
        pylab.ylabel(r'$y$',fontsize = 14)

        pylab.title('2D Surface',fontsize = 14)
        pylab.savefig('../plots/2d.png')
        pylab.show()
