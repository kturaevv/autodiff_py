import math
from functools import lru_cache


@lru_cache(maxsize=None)
def tanh(a):
    e_to_2a = math.exp(2 * a)
    return (e_to_2a - 1) / (e_to_2a + 1)
