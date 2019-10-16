#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from setuptools import setup, Extension
from setuptools import find_packages

NAME            = 'ztlearn'
DESCRIPTION     = 'Minimalistic Python Machine Learning Toolkit.'
URL             = 'https://github.com/jefkine/zeta-learn'
DOWLOAD_URL     = 'https://github.com/jefkine/zeta-learn/archive/master.zip'
EMAIL           = 'jefkine@gmail.com'
AUTHOR          = 'Jefkine Kafunah'
REQUIRES_PYTHON = '>=3.5.0'
VERSION         = None
REQUIRED        = ['numpy', 'matplotlib', 'scipy']

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(
    name                          = NAME,
    version                       = about['__version__'],
    description                   = DESCRIPTION,
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    author                        = AUTHOR,
    author_email                  = EMAIL,
    python_requires               = REQUIRES_PYTHON,
    url                           = URL,
    install_requires              = REQUIRED,
    include_package_data          = True,
    license                       = 'MIT',
    classifiers                   = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    packages             = find_packages(exclude=('docs',))
)
