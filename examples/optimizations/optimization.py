# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""

For the function y = theta^2,  we can plot the y  against the  thetas and we can
see that for the  range -5 to 6 with a step of 1, we  bypass the minima at (0,0),
beyond which we start rising yet again.

The np.meshgrid function helps us to merge the ys and the thetas so thatthe plot
that  follows is a  combination of  the function (curve) and  the evaluations at
various points of y (the red dots)

Our next task is then to show that  we can use methods like SGD and Momentum SGD
to find the minimum with ease

"""

f = lambda theta: theta**2

theta = np.arange(-5, 6, 1)
y     = f(theta)

thetas, ys = np.meshgrid(theta, y, sparse = True)

plt.figure(figsize=(6, 5))
plt.plot(theta, y)

plt.scatter(thetas, ys, color='r')

plt.xlabel(r'$\theta$',fontsize = 18)
plt.ylabel(r'$y$',fontsize = 18)

plt.show()
