import numpy as np
import matplotlib.pyplot as plt


def wave(data, title,
         ts=None, tsfmt=None,
         overlay=True,
         xlabel=None, ylabel=None,
         pointidx=None, fillidx=None,
         savefile=None, figsize=(10, 2)):
    if len(data.shape) > 1:
        T, F = data.shape
        if overlay:
            plt.figure(figsize=figsize)
            for i in range(F):
                plt.plot(data[:, i])
            plt.title(title)
            if pointidx is not None:
                for p in pointidx:
                    plt.plot(p, data[p, 0], 'ro')
        else:
            f, ax = plt.subplots(F, 1, sharex=True,
                                 figsize=(figsize[0], figsize[1]*F))
            if F == 1:
                ax = [ax]
            for i in range(F):
                ax[i].plot(data[:, i])
                ax[i].set_title(title[i])
                if pointidx is not None:
                    for p in pointidx:
                        ax[i].plot(p, data[p, i], 'ro')
    else:
        T = data.shape[0]
        plt.figure(figsize=figsize)
        plt.plot(data)
        # plt.axis('tight') # for matplotlib 1.5
        plt.title(title)
        if pointidx is not None:
            for p in pointidx:
                plt.plot(p, data[p], 'ro')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if ts is not None:
        ticks = [x.astype(int) for x in plt.gca().get_xticks()]
        if ticks[-1] > T:
            ticks[-1] = T-1
        if tsfmt is not None:
            from datetime import datetime
            fmt = '%Y-%m-%d %H:%M'
            tick_labels = [datetime.strptime(ts[x], fmt).strftime(tsfmt)
                           for x in ticks]
        else:
            tick_labels = [ts[x][:7] for x in ticks]
        plt.xticks(ticks, tick_labels)
    if savefile is not None:
        plt.savefig(savefile)
    if fillidx is not None:
        ylim = plt.gca().get_ylim()
        if len(data.shape) > 1:
            if overlay:
                plt.fill_between(range(T), ylim[0], ylim[1],
                                 where=fillidx, edgecolor='None',
                                 facecolor='red', alpha=0.2,
                                 interpolate=False)
                plt.gca().set_ylim(ylim)
            else:
                for i in range(F):
                    ylim = ax[i].get_ylim()
                    ax[i].fill_between(range(T), ylim[0], ylim[1],
                                       where=fillidx, edgecolor='None',
                                       facecolor='red', alpha=0.2,
                                       interpolate=False)
                    ax[i].set_ylim(ylim)
        else:
            plt.fill_between(range(T), ylim[0], ylim[1],
                             where=fillidx, edgecolor='None',
                             facecolor='red', alpha=0.2,
                             interpolate=False)
            plt.gca().set_ylim(ylim)
    plt.show()
    plt.close()


def abnormal(data, title, ts, events, threshold, cont=12):
    from .utils import getRanges
    canidx = data > threshold
    if cont > 0:
        for (i, j) in getRanges(np.where(canidx)[0]):
            if j-i < cont:
                canidx[i:j+1] = False
            else:
                print("%s ~ %s" % (ts[i], ts[j]))
    wave(data=data, title=title, ts=ts,
         ylabel='Degree of Abnormality',
         figsize=(10, 3),
         pointidx=[np.where(ts == e)[0][0] for e in events],
         fillidx=canidx)
