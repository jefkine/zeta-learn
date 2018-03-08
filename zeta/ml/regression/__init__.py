# -*- coding: utf-8 -*-

# import file(s)
from . import base
from . import linear
from . import logistic
from . import polynomial
from . import elasticnet

# base model
from .base import Regression

# linear model
from .linear import LinearRegression

# logistic model
from .logistic import LogisticRegression

# polynomial model
from .polynomial import PolynomialRegression

# elastic net model
from .elasticnet import ElasticNetRegression
