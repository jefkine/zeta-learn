# -*- coding: utf-8 -*-

import time
import types
from functools import wraps

class LogIfBusy:

    def __init__(self, func):
        wraps(func)(self)

    def __call__(self, *args, **kwargs):        
        print('\nSTART: {}\n'.format(time.strftime("%a, %d %b %Y %H:%M:%S")))
        start = time.time()        
        result = self.__wrapped__(*args, **kwargs)        
        stop = time.time()
        print('\nFINISH: {}\n'.format(time.strftime("%a, %d %b %Y %H:%M:%S")))        
        print('TIMER: {} operation took: {:2.4f} seconds to complete.\n'.format(self.__wrapped__.__name__, stop-start))
        return result

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)
