from __future__ import absolute_import
from __future__ import division

from .utils import subplots
from .utils import makeKwargs
from .utils import closeWithSave

import numpy as np
import matplotlib.pyplot as plt


@closeWithSave
def histAll(data, title=None, bins=None,
            color='#c5c5c5', edgecolor='None',
            xlabel=None, ylabel=None,
            savefile=None, grid=(None, 5), figsize=(10, 4)):
    grid, axarr, _ = subplots(data.shape[1], grid, figsize)

    kwargs = makeKwargs(bins=bins,
                        colors=color,
                        edgecolor=edgecolor)

    for i in range(data.shape[1]):
        ax = axarr[i // grid[1], i % grid[1]]
        ax.hist(data[~np.isnan(data[:, i]), i], **kwargs)

        if title is not None:
            ax.set_title(title[i])

        if xlabel is None:
            plt.setp(ax.get_xticklabels(), visible=False)

        if ylabel is None:
            plt.setp(ax.get_yticklabels(), visible=False)

    plt.tight_layout()


@closeWithSave
def corr(data, xlabel=None, ylabel=None,
         title=None, colorbar=True,
         window=None, sample=1000,
         savefile=None, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    if window is not None:
        corrcoef = []
        for s in range(sample):
            i = np.random.randint(0, data.shape[0]-window)
            c = np.corrcoef(data[i:(i+window)].T)
            corrcoef.append(c)
        results = np.nanmean(np.array(corrcoef), axis=0)
    else:
        results = np.corrcoef(data[np.sum(np.isnan(data), axis=1) == 0].T)
    plt.imshow(results)

    if title is not None:
        plt.title(title)

    if xlabel is None:
        plt.setp(plt.gca().get_xticklabels(), visible=False)

    if ylabel is None:
        plt.setp(plt.gca().get_yticklabels(), visible=False)
    else:
        yticks = [x.astype(int) for x in plt.gca().get_yticks()]
        if yticks[0] < 0:
            yticks = np.delete(yticks, 0)
        if yticks[-1] >= len(ylabel):
            yticks = np.delete(yticks, -1)
        plt.gca().set_yticks(yticks)
        plt.gca().set_yticklabels(ylabel[yticks])

    if colorbar:
        plt.colorbar()

    return results


@closeWithSave
def dualBar(data1, data2,
            color1='#999999',
            title=None, xlabel=None,
            ylabel1=None, ylabel2=None, width=3,
            savefile=None, figsize=(8, 3)):
    plt.figure(figsize=figsize)
    ax1 = plt.gca()
    ax2 = plt.gca().twinx()

    K = data1.shape[0]
    ax1.bar(width*np.arange(K)-width*0.2, np.mean(data1, axis=1), width*0.35,
            yerr=np.std(data1, axis=1), color=color1)
    ax2.bar(width*np.arange(K)+width*0.2, np.mean(data2, axis=1), width*0.35,
            yerr=np.std(data2, axis=1))

    if ylabel1 is not None:
        ax1.set_ylabel(ylabel1, color='gray')
    if ylabel2 is not None:
        ax2.set_ylabel(ylabel2)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xticks(width*np.arange(K), xlabel, rotation='vertical')
    plt.xlim([-width, width*K])


@closeWithSave
def multiBar(data, savefile=None, grid=(1, None),
             figsize=(8, 2), sharey=False):
    grid, axarr, _ = subplots(len(data), grid, figsize, sharey)

    for (i, d) in enumerate(data):
        ax = axarr[i // grid[1], i % grid[1]]
        ax.bar(range(len(d)), d)


@closeWithSave
def scatter(x, y, xlabel=None, ylabel=None,
            title=None, suptitle=None,
            identityline=False, markersize=1,
            cmap='rainbow', colorbar=False, colorbar_labels=None,
            savefile=None, grid=(1, None), figsize=(8, 3)):
    grid, axarr, f = subplots(len(x), grid, figsize)

    colors = np.linspace(0, 1, len(x[0]))

    if suptitle is not None:
        plt.suptitle(suptitle)

    for i in range(len(x)):
        ax = axarr[i // grid[1], i % grid[1]]
        if identityline:
            lim = (min(min(x[i]), min(y[i])), max(max(x[i]), max(y[i])))
            ax.plot(lim, lim, 'k--', color='gray')
        sc = ax.scatter(x[i], y[i], c=colors,
                        cmap=cmap, s=markersize)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title[i])

    if colorbar:
        cbar = f.colorbar(sc)
        if colorbar_labels is not None:
            yticks = cbar.ax.get_yticks()
            ytlabels = colorbar_labels[::len(colors) // (len(yticks) - 1) - 1]
            cbar.set_ticks(yticks)
            cbar.set_ticklabels(ytlabels)


@closeWithSave
def hist(data, bins=None,
         labels=None, colors=None,
         xlabel=None, ylabel=None,
         title=None, suptitle=None,
         savefile=None, grid=(1, None),
         figsize=(12, 3), sharey=True):
    grid, axarr, _ = subplots(len(data), grid, figsize, sharey)

    if suptitle is not None:
        plt.suptitle(suptitle)

    for i in range(len(data)):
        ax = axarr[i // grid[1], i % grid[1]]
        if isinstance(data[i], list):
            for j in range(len(data[i])):
                kwargs = makeKwargs(idx=j,
                                    bins=bins,
                                    labels=labels,
                                    colors=colors)
                ax.hist(data[i][j], **kwargs)
        else:
            ax.hist(data[i])
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if i == 0:
            ax.legend()
            if ylabel is not None:
                ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title[i])


@closeWithSave
def multiPredRange(key, actual, predicted,
                   xlabel, ylabel,
                   savefile=None, grid=(1, None), figsize=(12, 3)):
    grid, axarr, _ = subplots(len(key), grid, figsize, sharey=True)

    for (i, k) in enumerate(key):
        ax = axarr[i // grid[1], i % grid[1]]
        idx = np.argsort(actual[k])
        ax.plot(actual[k][idx], label='Actual',
                alpha=0.7, color='red')
        ax.plot(np.mean(predicted[k], axis=1)[idx], label='Predicted',
                linestyle='', marker='o', markersize=1)
        ax.fill_between(range(len(idx)),
                        np.min(predicted[k], axis=1)[idx],
                        np.max(predicted[k], axis=1)[idx],
                        alpha=0.5, edgecolor=None)
        ax.set_title(k)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if i == 0:
            ax.legend()
            if ylabel is not None:
                ax.set_ylabel(ylabel)
