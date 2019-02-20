# -*- coding: utf-8 -*-

import numpy as np

#-----------------------------------------------------------------------------#
#                       TEXT UTILITY FUNCTIONS                                #
#-----------------------------------------------------------------------------#

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


def pad_sequence(sequence,
                           maxlen     = None,
                           dtype      = 'int32',
                           padding    = 'pre',
                           truncating = 'pre',
                           value      = 0.0):
    """ pad or truncate sequences depending on size of maxlen - longest sequence """
    if (maxlen is None):  return sequence

    np_sequence   = np.array(sequence)
    sequence_size = np_sequence.size

    if (maxlen > sequence_size):
        pad_size = maxlen - sequence_size
        if (padding == 'pre'):  front_pad, back_pad = pad_size, 0
        if (padding == 'post'): front_pad, back_pad = 0, pad_size

        paded = np.pad(sequence,
                                 (front_pad, back_pad),
                                 'constant',
                                 constant_values = (value, value))
        return paded

    if (sequence_size > maxlen):
        trunc_size = sequence_size - maxlen
        if (truncating == 'pre'):  truncated = np_sequence[:-trunc_size]
        if (truncating == 'post'): truncated = np_sequence[trunc_size:]

        return truncated

    if(sequence_size == maxlen): return np_sequence


def get_sentence_tokens(text_list, maxlen = None, dtype = 'int32'):
    unique_words = list(set(text_list.split())) # get the unique words

    import re
    sentences = re.split(r'(?<=[.?!])\s+', text_list)

    if maxlen is None:
        maxlen = max(longest_sentence(sentences))

    sentence_index_list = []
    for _, s in enumerate(sentences):
        sentence_index = list(map(unique_words.index, s.split()))
        padded_index   = pad_sequence(sentence_index, maxlen)
        sentence_index_list.append(padded_index)

    return np.array(sentence_index_list), len(unique_words), maxlen


def longest_sentence(sentences):
    """ find longest sentence in a list of sentences """
    if isinstance(sentences, list):
        yield len(sentences)
        for sentence in sentences:
            yield from longest_sentence(sentence)
