# -*- coding: utf-8 -*-

import numpy as np

from ztlearn.optimizers import register_opt
from ztlearn.toolkit import GbOptimization as gbopt


"""2D module"""

# def f(x):
#    return x[0]**2
#
# def df(x):
#    return np.array([2*x[0]])
#
# opt   = register_opt(optimizer_name = 'sgd', momentum = 0.1, lr = 0.1)
# optim = gbopt(optimizer = opt, init_method = 'he_normal')
# optim.run(f, df, params = 1, epochs = 100)
# optim.plot_2d(f)

"""3D module"""

'''
def f2(x):
    return x[0]**2 + x[1]**2

def df2(x):
    return np.array([2*x[0], 2*x[1]])
'''

def f2(x):
    return 2*x[0]**3 + x[1]**4

def df2(x):
    return np.array([6*x[0]**2, 4*x[1]**3])

opt   = register_opt(optimizer_name = 'sgd_momentum', momentum = 0.01, lr = 0.001)
optim = gbopt(optimizer = opt, init_method = 'he_normal')
optim.run(f2, df2, params = 2, epochs = 1500)
optim.plot_3d(f2)
