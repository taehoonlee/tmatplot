import numpy as np


def getRanges(data):
    from operator import itemgetter
    from itertools import groupby
    ranges = []
    for k, g in groupby(enumerate(data), lambda (i, x): i-x):
        group = map(itemgetter(1), g)
        ranges.append((group[0], group[-1]))
    return ranges
