from __future__ import absolute_import

from .utils import subplots
from .utils import getRanges
from .utils import makeKwargs
from .utils import closeWithSave

import numpy as np
import matplotlib.pyplot as plt


@closeWithSave
def wave(data, title=None,
         ts=None, tsfmt=None,
         colors=None, overlay=True,
         xlabel=None, ylabel=None,
         pointidx=None,
         fillidx=None, fillcolor='red',
         savefile=None, close=True, figsize=(10, 2),
         sharex=True, tight=False):
    if data.ndim == 1:
        data = np.expand_dims(data, -1)
    T, F = data.shape
    if overlay:
        plt.figure(figsize=figsize)
        if F > 1:
            for i in range(F):
                plt.plot(data[:, i])
        else:
            plt.plot(data)
        plt.title(title)
        if pointidx is not None:
            for p in pointidx:
                if F > 1:
                    plt.plot(p, data[p, 0], 'ro')
                else:
                    plt.plot(p, data[p], 'ro')
    else:
        figsize = (figsize[0], F * figsize[1])
        _, axarr, _ = subplots(F, (None, 1), figsize=figsize, sharex=sharex)
        for i in range(F):
            ax = axarr[0, i]
            kwargs = makeKwargs(idx=i, colors=colors)
            ax.plot(data[:, i], **kwargs)
            if title is not None:
                ax.set_title(title[i])
            if pointidx is not None:
                for p in pointidx:
                    ax.plot(p, data[p, i], 'ro')

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if ts is not None:
        ticks = [x.astype(int) for x in plt.gca().get_xticks()]
        if ticks[-1] >= T:
            ticks = np.delete(ticks, -1)
        if tsfmt is not None:
            from datetime import datetime
            fmt = '%Y-%m-%d %H:%M'
            tick_labels = [datetime.strptime(ts[x], fmt).strftime(tsfmt)
                           for x in ticks]
        else:
            tick_labels = [ts[x][:7] for x in ticks]
        plt.xticks(ticks, tick_labels)

    if fillidx is not None:
        if overlay:
            ylim = plt.gca().get_ylim()
            plt.fill_between(range(T), ylim[0], ylim[1],
                             where=fillidx, edgecolor='None',
                             facecolor=fillcolor, alpha=0.2,
                             interpolate=False)
            plt.gca().set_ylim(ylim)
        else:
            for i in range(F):
                ax = axarr[0, i]
                ylim = ax.get_ylim()
                ax.fill_between(range(T), ylim[0], ylim[1],
                                where=fillidx, edgecolor='None',
                                facecolor=fillcolor, alpha=0.2,
                                interpolate=False)
                ax.set_ylim(ylim)

    if tight is True:
        plt.tight_layout()


def abnormal(data, title, ts, events, threshold, cont=12, tsfmt=None):
    canidx = data > threshold
    if cont > 0:
        for (i, j) in getRanges(np.where(canidx)[0]):
            if j-i < cont:
                canidx[i:j+1] = False
            else:
                print("%s ~ %s" % (ts[i], ts[j]))
    wave(data=data, title=title, ts=ts, tsfmt=tsfmt,
         ylabel='Degree of Abnormality',
         figsize=(10, 3),
         pointidx=[np.where(ts == e)[0][0] for e in events],
         fillidx=canidx)
