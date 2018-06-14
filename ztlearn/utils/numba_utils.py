# -*- coding: utf-8 -*-

from functools import wraps

def jit(*jit_args, **jit_kwargs):
    def inner_decorator(function):
        @wraps(function)
        def inner_wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            return result
        return inner_wrapper
    return inner_decorator

use_numba = False
