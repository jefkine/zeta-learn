# -*- coding: utf-8 -*-

import numpy as np

def get_pad(padding, input_height, input_width, stride_height, stride_width, kernel_height, kernel_width):
    if padding == 'valid':
        return (0, 0), (0, 0)

    elif padding == 'same':
        if (input_height % stride_height == 0):
          pad_along_height = max(kernel_height - stride_height, 0)
        else:
          pad_along_height = max(kernel_height - (input_height % stride_height), 0)

        if (input_width % stride_width == 0):
          pad_along_width = max(kernel_width - stride_width, 0)
        else:
          pad_along_width = max(kernel_width - (input_width % stride_width), 0)

        pad_top    = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left   = pad_along_width // 2
        pad_right  = pad_along_width - pad_left

        return (pad_top, pad_bottom), (pad_left, pad_right)

# Original Code: CS231n Stanford  http://cs231n.github.io/assignments2017/assignment2/
def get_im2col_indices(x_shape, field_height = 3, field_width = 3, padding = ((0, 0), (0, 0)), stride = 1):
    # First figure out what the size of the output should be
    N, C, H, W            = x_shape
    pad_height, pad_width = padding

    assert (H + np.sum(pad_height) - field_height) % stride == 0
    assert (W + np.sum(pad_width)  - field_height) % stride == 0

    out_height = (H + np.sum(pad_height) - field_height) / stride + 1
    out_width  = (W + np.sum(pad_width)  - field_width)  / stride + 1

    i0 = np.repeat(np.arange(field_height, dtype = 'int32'), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height, dtype = 'int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width, dtype = 'int32'), int(out_height))
    i  = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j  = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C, dtype='int32'), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding, stride = 1):
    """ An implementation of im2col based on some fancy indexing """
    pad_height, pad_width = padding

    x_padded = np.pad(x, ((0, 0), (0, 0), pad_height, pad_width), mode = 'constant')
    k, i, j  = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols     = x_padded[:, k, i, j]
    C        = x.shape[1]
    cols     = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

    return cols

def col2im_indices(cols, x_shape, field_height = 3, field_width = 3, padding = ((0, 0), (0, 0)), stride = 1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W            = x_shape
    pad_height, pad_width = padding
    H_padded, W_padded    = H + np.sum(pad_height), W + np.sum(pad_width)

    x_padded = np.zeros((N, C, H_padded, W_padded), dtype = cols.dtype)
    k, i, j  = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    pad_size = (np.sum(pad_height)/2).astype(int)
    if pad_size == 0:
        return x_padded
    return x_padded[:, :, pad_size:-pad_size, pad_size:-pad_size]

pass
