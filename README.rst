zeta-learn
----------
zeta-learn is a minimalistic python machine learning library designed to deliver
fast and easy model prototyping.

zeta-learn aims to provide an extensive understanding of machine learning through
the use of straightforward algorithms and readily implemented examples making
it a useful resource for researchers and students.

 * **Documentation:** https://zeta-learn.com
 * **Python versions:** 3.5 and above
 * **Free software:** MIT license

Dependencies
------------
 - numpy >= 1.15.0
 - scikit-learn >= 0.18.0
 - matplotlib >= 2.0.0

Features
--------
 - Keras like Sequential API for building models.
 - Built on Numpy and Matplotlib.
 - Examples folder with readily implemented machine learning models.

Install
-------
  - pip install ztlearn

Examples
--------

Convolutional Neural Network (CNN)
##################################

.. ::

  DIGITS CNN

   Input Shape: (1, 8, 8)
  +---------------------+--------+--------------+
  ¦ LAYER TYPE          ¦ PARAMS ¦ OUTPUT SHAPE ¦
  +---------------------+--------+--------------+
  ¦ Conv2D              ¦    320 ¦   (32, 8, 8) ¦
  ¦ Activation: RELU    ¦      0 ¦   (32, 8, 8) ¦
  ¦ Dropout             ¦      0 ¦   (32, 8, 8) ¦
  ¦ BatchNormalization  ¦   4096 ¦   (32, 8, 8) ¦
  ¦ Conv2D              ¦  18496 ¦   (64, 8, 8) ¦
  ¦ Activation: RELU    ¦      0 ¦   (64, 8, 8) ¦
  ¦ MaxPooling2D        ¦      0 ¦   (64, 7, 7) ¦
  ¦ Dropout             ¦      0 ¦   (64, 7, 7) ¦
  ¦ BatchNormalization  ¦   6272 ¦   (64, 7, 7) ¦
  ¦ Flatten             ¦      0 ¦      (3136,) ¦
  ¦ Dense               ¦ 803072 ¦       (256,) ¦
  ¦ Activation: RELU    ¦      0 ¦       (256,) ¦
  ¦ Dropout             ¦      0 ¦       (256,) ¦
  ¦ BatchNormalization  ¦    512 ¦       (256,) ¦
  ¦ Dense               ¦   2570 ¦        (10,) ¦
  +---------------------+--------+--------------+

  TOTAL PARAMETERS: 835338

.. image:: /examples/plots/results/cnn/digits_cnn_tiled_results.png
      :align: center
      :alt: digits cnn results tiled
