from __future__ import absolute_import
from __future__ import division

from .styles import get_colors
from .utils import subplots
from .utils import makeKwargs
from .utils import closeWithSave

import numpy as np
import matplotlib.pyplot as plt


@closeWithSave
def corr(data, xlabel=None, ylabel=None,
         title=None, colorbar=True,
         window=None, sample=1000,
         savefile=None, close=True, figsize=(8, 6)):
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
            color1='#999999', color2='C0',
            title=None, xlabel=None,
            ylabel1=None, ylabel2=None, width=3,
            savefile=None, close=True, figsize=(8, 3)):
    plt.figure(figsize=figsize)

    K = data1.shape[0]
    for (i, (ax, data, color, ylabel)) in \
        enumerate(zip([plt.gca(), plt.gca().twinx()],
                      [data1, data2], [color1, color2], [ylabel1, ylabel2])):
        ax.bar(width * (np.arange(K) + 0.4 * (i - 0.5)),
               np.mean(data, axis=1),
               width * 0.35,
               yerr=np.std(data, axis=1),
               color=color)

        if ylabel is not None:
            ax.set_ylabel(ylabel, color=color)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xticks(width*np.arange(K), xlabel, rotation='vertical')
    plt.xlim([-width, width*K])


@closeWithSave
def multiBar(data, xlabel=None, ylabel=None,
             title=None, suptitle=None,
             colors='C0', alphas=None, edgecolors=None,
             savefile=None, close=True,
             grid=(1, None), figsize=(8, 2),
             sharey=False, singleconf=False, tight=True):
    grid, axarr, _ = subplots(len(data), grid, figsize, sharey=sharey)

    if suptitle is not None:
        plt.suptitle(suptitle)

    for (i, d) in enumerate(data):
        ax = axarr[i // grid[1], i % grid[1]]
        kwargs = makeKwargs(idx=i,
                            colors=colors,
                            alphas=alphas,
                            edgecolors=edgecolors,
                            singleconf=singleconf)
        ax.bar(range(len(d)), d, **kwargs)

        if isinstance(xlabel, list):
            ax.set_xlabel(xlabel[i])
        elif (xlabel is not None) and (i == 0):
            ax.set_xticks(range(len(xlabel)))
            ax.set_xticklabels(xlabel, rotation='45')

        if isinstance(ylabel, list):
            ax.set_xlabel(ylabel[i])
        elif (ylabel is not None) and (i == 0):
            ax.set_ylabel(ylabel)

        if isinstance(title, list):
            ax.set_title(title[i])
        elif title is not None:
            ax.set_title(title)

    if tight is True:
        plt.tight_layout()


@closeWithSave
def scatter(x, y, c=None, xlabel=None, ylabel=None,
            title=None, suptitle=None,
            identityline=False, marker=None, markersize=1,
            cmap='rainbow', alpha=None, labels=None,
            colorbar=False, colorbar_labels=None,
            savefile=None, close=True,
            grid=(1, None), figsize=(8, 3), tight=False):
    if not isinstance(x, list):
        x = [x]
    if not isinstance(y, list):
        y = [y]
    grid, axarr, f = subplots(len(x), grid, figsize)

    if suptitle is not None:
        plt.suptitle(suptitle)

    for (i, (_x, _y)) in enumerate(zip(x, y)):
        ax = axarr[i // grid[1], i % grid[1]]
        if identityline:
            lim = (min(min(_x), min(_y)), max(max(_x), max(_y)))
            ax.plot(lim, lim, 'k--', color='gray')
        if (c is None) and (cmap is not None):
            c = np.linspace(0, 1, len(_x))
        kwargs = makeKwargs(idx=i, alphas=alpha, markers=marker)
        kwargs['s'] = markersize
        if cmap is not None:
            kwargs['cmap'] = cmap
        if c is not None:
            kwargs['c'] = c
        if labels is not None:
            kwargs.pop('c')
            for (j, label) in enumerate(labels):
                sc = ax.scatter(_x[c == j], _y[c == j], label=label, **kwargs)
        else:
            sc = ax.scatter(_x, _y, **kwargs)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title[i])

    if labels is not None:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2)

    if colorbar:
        cbar = f.colorbar(sc)
        if colorbar_labels is not None:
            yticks = cbar.ax.get_yticks()
            ytlabels = colorbar_labels[::len(c) // (len(yticks) - 1) - 1]
            cbar.set_ticks(yticks)
            cbar.set_ticklabels(ytlabels)

    if tight is True:
        plt.tight_layout()


@closeWithSave
def hist(data, bins=None, labels=None,
         groupby=None, density=False,
         colors=None, alphas=None, edgecolors=None,
         xlabel=None, ylabel=None,
         title=None, suptitle=None,
         savefile=None, close=True,
         grid=(1, None), figsize=(12, 3),
         sharey=True, tight=False):
    if groupby is not None:
        _bins = bins
        bins = []
        for i in range(data.shape[1]):
            _, b = np.histogram(data[:, i], bins=_bins)
            bins.append(b)
        groups = np.unique(groupby)
        data = [[data[groupby == g, i] for g in groups]
                for i in range(data.shape[1])]
        if colors is None:
            colors = get_colors(1)
            if len(groups) <= len(colors) // 2:
                colors = colors[::2]
            colors = ['#%s' % c for c in colors]

    grid, axarr, _ = subplots(len(data), grid, figsize, sharey=sharey)

    if suptitle is not None:
        plt.suptitle(suptitle)

    for (i, d) in enumerate(data):
        ax = axarr[i // grid[1], i % grid[1]]
        if isinstance(d, list):
            for j in range(len(d)):
                _bins = bins[i] if groupby is not None else bins
                kwargs = makeKwargs(idx=j,
                                    bins=_bins,
                                    labels=labels,
                                    colors=colors,
                                    alphas=alphas,
                                    edgecolors=edgecolors)
                if density is True:
                    kwargs['weights'] = np.ones_like(d[j]) / d[j].shape[0]
                ax.hist(d[j], **kwargs)
        else:
            kwargs = makeKwargs(idx=i,
                                bins=bins,
                                colors=colors,
                                alphas=alphas,
                                edgecolors=edgecolors)
            if density is True:
                kwargs['weights'] = np.ones_like(d[j]) / d[j].shape[0]
            ax.hist(d[~np.isnan(d)], **kwargs)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        if sharey is True:
            if i == 0:
                ax.legend()
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
        else:
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

        if title is not None:
            ax.set_title(title[i])

    if tight is True:
        plt.tight_layout()


def histAll(data, title=None, bins=None,
            color='#c5c5c5', edgecolor='None',
            xlabel=None, ylabel=None,
            savefile=None, close=True,
            grid=(None, 5), figsize=(10, 4)):
    listdata = [data[:, i] for i in range(data.shape[1])]
    hist(listdata, title=title, bins=bins,
         colors=color, edgecolors=edgecolor,
         xlabel=xlabel, ylabel=ylabel,
         savefile=savefile, grid=grid, figsize=figsize,
         sharey=False, tight=True)


@closeWithSave
def multiPredRange(key, actual, predicted,
                   xlabel, ylabel,
                   savefile=None, close=True,
                   grid=(1, None), figsize=(12, 3),
                   sharey=True, tight=True):
    grid, axarr, _ = subplots(len(key), grid, figsize, sharey=sharey)

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

    if tight is True:
        plt.tight_layout()
