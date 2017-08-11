from __future__ import absolute_import
from __future__ import division

from .utils import makeGrid
from .utils import makeArray
from .utils import closeWithSave

import numpy as np
import matplotlib.pyplot as plt


@closeWithSave
def histAll(data, title=None, bins=None,
            color='#c5c5c5', edgecolor='None',
            xlabel=None, ylabel=None,
            savefile=None, grid=(None, 5), figsize=(10, 4)):
    K = data.shape[1]
    grid = makeGrid(grid, K)
    f, axarr = plt.subplots(grid[0], grid[1], figsize=figsize)
    axarr = makeArray(axarr)

    for k in range(K):
        ax = axarr[k // grid[1], k % grid[1]]

        kwargs = {}
        if bins is not None:
            kwargs['bins'] = bins

        if color is not None:
            if isinstance(color, list):
                kwargs['color'] = color[k]
            else:
                kwargs['color'] = color

        if edgecolor is not None:
            kwargs['edgecolor'] = edgecolor

        ax.hist(data[~np.isnan(data[:, k]), k], **kwargs)

        if title is not None:
            ax.set_title(title[k])

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
def multiBar(data, savefile=None, grid=(1, None), figsize=(8, 2)):
    grid = makeGrid(grid, len(data))
    f, axarr = plt.subplots(grid[0], grid[1], figsize=figsize)
    axarr = makeArray(axarr, 1)

    for (k, d) in enumerate(data):
        ax = axarr[k]
        ax.bar(range(len(d)), d)


@closeWithSave
def scatter(x, y, xlabel=None, ylabel=None,
            title=None, suptitle=None,
            identityline=False, markersize=1,
            cmap='rainbow', colorbar=False, colorbar_labels=None,
            savefile=None, grid=(1, None), figsize=(8, 3)):
    K = len(x)
    colors = np.linspace(0, 1, len(x[0]))
    grid = makeGrid(grid, K)
    f, axarr = plt.subplots(grid[0], grid[1], figsize=figsize)
    axarr = makeArray(axarr)

    if suptitle is not None:
        plt.suptitle(suptitle)

    for k in range(K):
        ax = axarr[k // grid[1], k % grid[1]]
        if identityline:
            ax.plot([-5, 5], [-5, 5], 'k--', color='gray')
        sc = ax.scatter(x[k], y[k], c=colors,
                        cmap=cmap, s=markersize)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title[k])

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
         savefile=None, grid=(1, None), figsize=(12, 3)):
    K = len(data)
    grid = makeGrid(grid, K)
    f, axarr = plt.subplots(grid[0], grid[1], figsize=figsize, sharey=True)
    axarr = makeArray(axarr)

    if suptitle is not None:
        plt.suptitle(suptitle)

    for k in range(K):
        ax = axarr[k // grid[1], k % grid[1]]
        if isinstance(data[k], list):
            for i in range(len(data[k])):
                kwargs = {}
                if bins is not None:
                    kwargs['bins'] = bins
                if labels is not None:
                    kwargs['label'] = labels[i]
                if colors is not None:
                    try:
                        kwargs['alpha'] = 0.2
                        kwargs['color'] = colors[i]
                    except:
                        kwargs['alpha'] = 0.5
                ax.hist(data[k][i], **kwargs)
        else:
            ax.hist(data[k])
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if k == 0:
            ax.legend()
            if ylabel is not None:
                ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title[k])


@closeWithSave
def multiPredRange(key, actual, predicted,
                   xlabel, ylabel,
                   savefile=None, grid=(1, None), figsize=(12, 3)):
    grid = makeGrid(grid, len(key))
    f, axarr = plt.subplots(grid[0], grid[1], figsize=figsize, sharey=True)
    axarr = makeArray(axarr, 1)
    for (i, k) in enumerate(key):
        ax = axarr[i]
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
