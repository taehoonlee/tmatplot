from __future__ import division

import numpy as np


def getRanges(data):
    from operator import itemgetter
    from itertools import groupby
    ranges = []
    for k, g in groupby(enumerate(data), lambda (i, x): i-x):
        group = map(itemgetter(1), g)
        ranges.append((group[0], group[-1]))
    return ranges


def makeArray(axarr, ndim=2):
    if isinstance(axarr, np.ndarray):
        if ndim == 2:
            if axarr.ndim == 1:
                return np.array([axarr])
            else:
                return axarr
        else:
            return axarr
    else:
        if ndim == 2:
            return np.array([[axarr]])
        else:
            return np.array([axarr])


def makeGrid(grid, objects):
    if grid[0] is None:
        return (((objects - 1) // grid[1]) + 1, grid[1])
    elif grid[1] is None:
        return (grid[0], ((objects - 1) // grid[0]) + 1)
    else:
        return (((objects - 1) // 5) + 1, 5)  # default is (x, 5)
