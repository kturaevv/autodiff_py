import math

from functools import lru_cache


@lru_cache(maxsize=None)
def tanh(a):
    return (math.exp(a) - math.exp(- a)) / (math.exp(a) + math.exp(-a))
