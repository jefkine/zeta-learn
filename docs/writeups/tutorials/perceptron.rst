The Perceptron
==============

The perceptron is a linear classifier used for binary classification. Learning is
achieved in a supervised setting (supervised learning) using stochastic gradient
descent. The percepton also features a threshold (activation) function which is
used to map the input between the required values.

zeta-learn has a simple ``Perceptron`` model that you can use to build up the
percepton as we will demonstrate below.

We shall use DIGITS, a dataset of handwritten digits with 1797 images. Each image
is represented by 8x8 pixels, with each containing a value 0 - 100 as its
grayscale value.

zeta-learn has a function which enables us to visualize samples of the DIGITS dataset

.. code-block:: python

    from sklearn import datasets
    from ztlearn.utils import plot_digits_img_results

    plot_digits_img_samples(datasets.load_digits())

The visuals generated from the above code snippet will look like the one below:

.. figure:: ../../img/digits_samples.png
    :width: 300px
    :align: center
    :height: 300px
    :alt: digits samples
    :figclass: align-center

Our full percepton model from zeta-learn will be

.. literalinclude:: ../../../examples/digits/digits_perceptron.py
   :language: python

The perceptron model attains 96.11% accuracy with the results obtained plotted below:

.. figure:: ../../img/digits_perceptron_results.png
    :width: 300px
    :align: center
    :height: 300px
    :alt: digits perceptron results
    :figclass: align-center

References:
  * `Perceptron Learning, Raul Rojas`_
  * `The Perceptron Algorithm, Avrim Blum (2010)`_
  * `Pen-Based Recognition of Handwritten Digits Data Set`_

.. _Perceptron Learning, Raul Rojas: https://page.mi.fu-berlin.de/rojas/neural/chapter/K4.pdf
.. _The Perceptron Algorithm, Avrim Blum (2010): https://www.cs.cmu.edu/~avrim/ML10/lect0125.pdf
.. _Pen-Based Recognition of Handwritten Digits Data Set: http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
