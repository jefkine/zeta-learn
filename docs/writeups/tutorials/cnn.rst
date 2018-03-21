Convolutional Neural Networks (CNNs)
====================================

Convolutional neural networks (CNNs) are a biologically-inspired variation of the
multilayer perceptrons (MLPs). Neurons in CNNs share weights unlike in MLPs where
each neuron has a separate weight vector. This sharing of weights ends up
reducing the overall number of trainable weights hence introducing sparsity.

zeta-learn utilizes the ``Sequential`` model api to help you build up a convolutional
neural network with ease as we will demonstrate below.

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

Our full CNN model from zeta-learn will be as follows

.. literalinclude:: ../../../examples/digits/digits_cnn.py
   :language: python

The CNN model attains 99.33% accuracy with the results obtained plotted below:

.. figure:: ../../img/digits_cnn_results.png
   :width: 300px
   :align: center
   :height: 300px
   :alt: digits cnn results
   :figclass: align-center

References:
  * `Convolutional Neural Networks UFLDL Tutorial`_
  * `Convolutional Neural Networks (LeNet)`_
  * `Backpropagation in Convolutional Neural Networks`_
  * `Convolutional Neural Networks (CNNs / ConvNets)`_

.. _Convolutional Neural Networks UFLDL Tutorial: http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/
.. _Convolutional Neural Networks (LeNet): http://deeplearning.net/tutorial/lenet.html
.. _Backpropagation in Convolutional Neural Networks: http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
.. _Convolutional Neural Networks (CNNs / ConvNets): http://cs231n.github.io/convolutional-networks/
