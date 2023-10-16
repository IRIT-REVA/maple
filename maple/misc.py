from typing import Sequence, Union
import numpy as np


def as_iterable(i: Union[int, Sequence[int]]):

    if not np.iterable(i):
        assert isinstance(i, int)
        i = [i]

    return i


def average_dict_values(d):

    return np.average(np.concatenate([v for v in d.values()]))


def variance_dict_values(d):

    return np.var(np.concatenate([v for v in d.values()]))


def consecutive_dict(a):

    return {i: v for i, (k, v) in enumerate(sorted(a.items()))}
