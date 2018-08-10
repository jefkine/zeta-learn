# -*- coding: utf-8 -*-

import numpy as np
from math import ceil

def alt_get_output_dims(input_height, input_width, kernel_size, strides, pad_height, pad_width):

    """
    FORMULA: [((W - KernelW + 2P) / Sw) + 1] and [((H - KernelH + 2P) / Sh) + 1]
    FORMULA: [((W - PoolW + 2P) / Sw) + 1] and [((H - PoolH + 2P) / Sh) + 1]
    """

    output_height = ((input_height - kernel_size[0] + np.sum(pad_height)) / strides[0]) + 1
    output_width  = ((input_width  - kernel_size[1] + np.sum(pad_width))  / strides[1]) + 1

    return output_height, output_width

def get_output_dims(input_height, input_width, kernel_size, strides, padding_type = 'valid'):

    """
    **SAME and VALID Padding**

    VALID: No padding is applied.  Assume that all  dimensions are valid so that input image
           gets fully covered by filter and stride you specified.

    SAME:  Padding is applied to input (if needed) so that input image gets fully covered by
           filter and stride you specified. For stride 1, this will ensure that output image
           size is same as input.

    References:
        [1] SAME and VALID Padding: http://bit.ly/2MtGgBM
    """

    if padding_type == 'same':
        output_height = ceil(float(input_height) / float(strides[0]))
        output_width  = ceil(float(input_width) / float(strides[1]))

    if padding_type == 'valid':
        output_height = ceil(float(input_height - kernel_size[0] + 1) / float(strides[0]))
        output_width  = ceil(float(input_width  - kernel_size[1] + 1) / float(strides[1]))

    return output_height, output_width

# unroll for toeplitz
def unroll_inputs(padded_inputs,
                                 batch_num,
                                 filter_num,
                                 output_height,
                                 output_width,
                                 kernel_size):

    unrolled_inputs = np.zeros((batch_num,
                                           filter_num,
                                           output_height * output_width,
                                           kernel_size**2))

    offset = 0
    for h in np.arange(output_height): # output height
        for w in np.arange(output_width): # output width
            for b in np.arange(batch_num): # batch number
                for f in np.arange(filter_num): # filter number
                     unrolled_inputs[b, f, offset, :] = padded_inputs[b,
                                                                      f,
                                                                      h:h+kernel_size,
                                                                      w:w+kernel_size].flatten()
            offset += 1

    return unrolled_inputs.reshape(filter_num * kernel_size**2, -1)
