
import numpy
# import scipy

from scipy.stats import norm
from math import sin, cos, exp, log, sqrt
import random

def sample(N, noise_sigma = 0.01):
    t  = 3 + 12 * numpy.random.random((N,))
    if noise_sigma > 0.0:
        ex = numpy.random.normal(0, noise_sigma, (N,))
        ey = numpy.random.normal(0, noise_sigma, (N,))
    else:
        ex = numpy.zeros((N,))
        ey = numpy.zeros((N,))

    X = 0.04*t*numpy.sin(t) + ex
    Y = 0.04*t*numpy.cos(t) + ey

    return (X,Y)

