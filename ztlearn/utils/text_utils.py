# -*- coding: utf-8 -*-

import numpy as np

from numba import jit
from numba import config
from .numba_utils import CACHE_FLAG
from .numba_utils import NOGIL_FLAG
from .numba_utils import DISABLE_JIT_FLAG

config.DISABLE_JIT = DISABLE_JIT_FLAG

#-----------------------------------------------------------------------------#
#                       TEXT UTILITY FUNCTIONS                                #
#-----------------------------------------------------------------------------#

# @@BUG: numba - internal error with dictionary composition https://github.com/numba/numba/issues/3028
# @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
def gen_char_sequence_xtym(text, maxlen, step, tensor_dtype = np.int):
    chars     = sorted(list(set(text)))
    len_chars = len(chars)
    len_text  = len(text)

    char_to_indices = {c: i for i, c in enumerate(chars)}
    # indices_to_char = {i: c for i, c in enumerate(chars)}

    sentences, next_chars = [], []
    for i in range(0, len_text - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    len_sentences = len(sentences)

    x = np.zeros((len_sentences, maxlen, len_chars), dtype = tensor_dtype)
    y = np.zeros((len_sentences, len_chars), dtype = tensor_dtype)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_to_indices[char]] = 1
        y[i, char_to_indices[next_chars[i]]] = 1

    return x, y, len_chars

# @@BUG: numba - internal error with dictionary composition https://github.com/numba/numba/issues/3028
# @jit(nogil = NOGIL_FLAG, cache = CACHE_FLAG)
def gen_char_sequence_xtyt(text, maxlen, step, tensor_dtype = np.int):
    chars     = sorted(list(set(text)))
    len_chars = len(chars)
    len_text  = len(text)

    char_to_indices = {c: i for i, c in enumerate(chars)}
    # indices_to_char = {i: c for i, c in enumerate(chars)}

    sentences, next_chars = [], []
    for i in range(0, len_text - maxlen + 1, step):
        sentences.append(text[i : i + maxlen])
        next_chars.append(text[i+1 : i+1 + maxlen])

    len_sentences = len(sentences)

    x = np.zeros((len_sentences, maxlen, len_chars), dtype = tensor_dtype)
    y = np.zeros((len_sentences, maxlen, len_chars), dtype = tensor_dtype)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_to_indices[char]] = 1

    for i, sentence in enumerate(next_chars):
        for t, char in enumerate(sentence):
            y[i, t, char_to_indices[char]] = 1

    return x, y, len_chars
