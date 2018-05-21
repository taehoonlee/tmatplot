from __future__ import absolute_import

from .styles import get_colors
from .utils import subplots
from .utils import getRanges
from .utils import closeWithSave

from time import mktime
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


@closeWithSave
def wave(data, title,
         ts=None, tsfmt=None,
         overlay=True,
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
            ax.plot(data[:, i])
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
        if ticks[-1] > T:
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
                ylim = ax[i].get_ylim()
                ax[i].fill_between(range(T), ylim[0], ylim[1],
                                   where=fillidx, edgecolor='None',
                                   facecolor=fillcolor, alpha=0.2,
                                   interpolate=False)
                ax[i].set_ylim(ylim)

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


@closeWithSave
def lotwave(data, title, suptitle=None,
            ts=None, tsfmt=None,
            meta=None, sampling=60, stepfilter=True,
            overlay=False,
            xlabel=None, ylabel=None,
            pointidx=None,
            fillidx=None, fillcolor='red',
            savefile=None, figsize=(10, 1),
            sharex=True, tight=True):
    if data.ndim == 1:
        data = np.expand_dims(data, -1)

    if data.shape[0] > 20000 and sampling > 1:
        data = data[::sampling]
        if ts is not None:
            ts = ts[::sampling]
        if meta is not None:
            meta = meta[::sampling]
    else:
        sampling = 1

    if meta is not None:
        step_id = meta[:, 3].astype(np.float32)
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
        if meta is not None:
            F += 1
        figsize = (figsize[0], F * figsize[1])
        _, axarr, _ = subplots(F, (None, 1), figsize=figsize, sharex=sharex)
        for i in range(F):
            ax = axarr[0, i]
            if i < data.shape[1]:
                plotdata = data[:, i]
                plottitle = title[i]
            else:
                plotdata = step_id
                plottitle = 'step_id'
            h, = ax.plot(plotdata, color='#999999')
            ax.text(0, np.nanmax(plotdata), plottitle, size=10,
                    bbox=dict(boxstyle='round',
                              ec=(0.6, 0.6, 0.8),
                              fc=(0.9, 0.9, 0.95)))

    if suptitle is not None:
        plt.suptitle(suptitle, y=1.02)

    handles = []
    legendlabels = []
    if meta is not None:
        lotid_change = np.concatenate([np.array([True]),
                                       meta[1:, 0] != meta[:-1, 0]])
        assert np.sum(lotid_change) == len(np.unique(meta[:, 0]))

        pointidx = np.where(lotid_change)[0]
        for i in range(F):
            ax = axarr[0, i]
            if i < F - 1:
                plotdata = data[:, i]
            else:
                plotdata = step_id
            for (j, p) in enumerate(pointidx):
                h, = ax.plot(p, plotdata[p],
                             color='#' + get_colors(1)[j % 16],
                             marker='o',
                             markersize=5,
                             linestyle='')
                if i == 0:
                    handles.append(h)
                    lot_id = meta[p, 0]
                    lot_idx = meta[:, 0] == lot_id
                    recipe_id = np.unique(meta[lot_idx, 1])
                    legendlabels.append("%s: %.1f m\n%s" %
                                        (lot_id,
                                         sum(lot_idx) * float(sampling) / 60.0,
                                         recipe_id))

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if ts is not None:
        ticks = [x.astype(int) for x in plt.gca().get_xticks()]
        if ticks[-1] >= T:
            ticks = np.delete(ticks, -1)
        if tsfmt is not None:
            try:
                fmt = '%Y-%m-%d %H:%M'
                tick_labels = [datetime.strptime(ts[x], fmt).strftime(tsfmt)
                               for x in ticks]
            except:
                fmt = '%Y-%m-%d %H:%M:%S.%f'
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
                ylim = ax[i].get_ylim()
                ax[i].fill_between(range(T), ylim[0], ylim[1],
                                   where=fillidx, edgecolor='None',
                                   facecolor=fillcolor, alpha=0.2,
                                   interpolate=False)
                ax[i].set_ylim(ylim)

    if tight is True:
        plt.tight_layout()

    if meta is not None:
        axarr[0, 0].legend(handles, legendlabels,
                           bbox_to_anchor=(1.02, 1), loc=2)
