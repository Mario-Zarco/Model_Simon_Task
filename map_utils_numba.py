import numpy as np
from numba import vectorize, float64


MIN_SEARCH_VALUE = -1.0
MAX_SEARCH_VALUE = 1.0

clipmin=-1.0e99
clipmax=1.0e99


@vectorize([float64(float64, float64, float64)])
def clip(x, min_, max_):
    if x > min_:
        temp = x
    else:
        temp = min_
    if temp < max_:
        return temp
    else:
        return max_


@vectorize([float64(float64, float64, float64)])
def map_search_parameters(x, min, max):
    m = (max - min) / (MAX_SEARCH_VALUE - MIN_SEARCH_VALUE)
    b = min - m * MIN_SEARCH_VALUE
    return clip(m * x + b, clipmin, clipmax)
