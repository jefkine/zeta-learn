# -*- coding: utf-8 -*-

import time
import types
from functools import wraps
from datetime import timedelta

class LogIfBusy:

    def __init__(self, func):
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        print('\nSTART: {}\n'.format(time.strftime("%a, %d %b %Y %H:%M:%S")))
        start  = time.time()
        result = self.__wrapped__(*args, **kwargs)
        stop   = time.time()
        print('\nFINISH: {}\n'.format(time.strftime("%a, %d %b %Y %H:%M:%S")))
        print('TIMER: {} operation took: {} (h:mm:ss) to complete.\n'.format(self.__wrapped__.__name__,
                                                                             timedelta(seconds = timedelta(seconds = (stop-start)).seconds)))

        return result

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)
            
