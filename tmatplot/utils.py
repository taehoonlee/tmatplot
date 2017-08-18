from __future__ import absolute_import
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def getRanges(data):
    from operator import itemgetter
    from itertools import groupby
    ranges = []
    for k, g in groupby(enumerate(data), lambda (i, x): i-x):
        group = map(itemgetter(1), g)
        ranges.append((group[0], group[-1]))
    return ranges


def makeKwargs(idx=None, bins=None, labels=None,
               colors=None, alpha=False, edgecolor=None):
    kwargs = {}
    if bins is not None:
        kwargs['bins'] = bins
    if labels is not None:
        kwargs['label'] = labels[idx]
    if colors is not None:
        if isinstance(colors, list):
            kwargs['color'] = colors[idx]
            if alpha:
                kwargs['alpha'] = 0.2
        else:
            kwargs['color'] = colors
            if alpha:
                kwargs['alpha'] = 0.5
    if edgecolor is not None:
        kwargs['edgecolor'] = edgecolor
    return kwargs


def subplots(num_objects, grid, figsize=None, sharey=None):
    if grid[0] is None:
        grid = (((num_objects - 1) // grid[1]) + 1, grid[1])
    elif grid[1] is None:
        grid = (grid[0], ((num_objects - 1) // grid[0]) + 1)
    else:
        grid = (((num_objects - 1) // 5) + 1, 5)  # default is (x, 5)

    kwargs = {}
    if figsize is not None:
        kwargs['figsize'] = figsize
    if sharey is not None:
        kwargs['sharey'] = sharey
    f, axarr = plt.subplots(grid[0], grid[1], **kwargs)

    if isinstance(axarr, np.ndarray):
        if axarr.ndim == 1:
            axarr = np.array([axarr])
    else:
        axarr = np.array([[axarr]])

    return grid, axarr, f


def closeWithSave(func):
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        savefile = kwargs.get('savefile', None)
        if savefile is not None:
            plt.savefig(savefile)
        plt.show()
        plt.close()
        return results
    return wrapper


def remove_utils(module_name):
    import sys
    from . import utils
    module = sys.modules[module_name]
    for util in dir(utils):
        if not util.startswith('_'):
            try:
                delattr(module, util)
            except:
                None
    delattr(module, 'mpl')
    delattr(module, 'utils')
    delattr(module, 'styles')
