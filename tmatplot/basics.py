from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def histAll(data, title=None, bins=100, color='#c5c5c5',
            savefile=None, grid=None, figsize=(10, 4)):
    K = data.shape[1]
    if grid is None:
        grid = (((K - 1) // 5) + 1, 5)

    f, axarr = plt.subplots(grid[0], grid[1], figsize=figsize)

    if isinstance(axarr, np.ndarray):
        if axarr.ndim == 1:
            axarr = np.array([axarr])
    else:
        axarr = np.array([[axarr]])

    for k in range(K):
        row = k // grid[1]
        col = k % grid[1]
        axarr[row, col].hist(data[~np.isnan(data[:, k]), k], bins,
                             facecolor=color, edgecolor='None')
        if title is not None:
            axarr[row, col].set_title(title[k])

    for i in axarr:
        for j in i:
            plt.setp(j.get_xticklabels(), visible=False)
            plt.setp(j.get_yticklabels(), visible=False)
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)

    plt.show()
    plt.close()


def corr(data, ylabel=None,
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

    if savefile is not None:
        plt.savefig(savefile)

    plt.show()
    plt.close()

    return results


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

    if savefile is not None:
        plt.savefig(savefile)

    plt.show()


def multiBar(data,
             savefile=None, grid=None,
             figsize=(8, 2)):
    if grid is None:
        grid = (1, len(data))

    f, axarr = plt.subplots(grid[0], grid[1], figsize=figsize)

    if not isinstance(axarr, np.ndarray):
        axarr = np.array([axarr])

    for (k, d) in enumerate(data):
        axarr[k].bar(range(len(d)), d)

    if savefile is not None:
        plt.savefig(savefile)

    plt.show()


def scatter(x, y, xlabel=None, ylabel=None,
            title=None, suptitle=None,
            identityline=False, markersize=1,
            cmap='rainbow', colorbar=False, colorbar_labels=None,
            savefile=None, grid=None, figsize=(8, 3)):
    K = len(x)
    colors = np.linspace(0, 1, len(x[0]))
    if grid is None:
        grid = (1, K)

    f, axarr = plt.subplots(grid[0], grid[1], figsize=figsize)

    if isinstance(axarr, np.ndarray):
        if axarr.ndim == 1:
            axarr = np.array([axarr])
    else:
        axarr = np.array([[axarr]])

    if suptitle is not None:
        plt.suptitle(suptitle)

    for k in range(K):
        row = k // grid[1]
        col = k % grid[1]
        if identityline:
            axarr[row, col].plot([-5, 5], [-5, 5], 'k--', color='gray')
        sc = axarr[row, col].scatter(x[k], y[k], c=colors,
                                     cmap=cmap, s=markersize)
        if xlabel is not None:
            axarr[row, col].set_xlabel(xlabel)
        if ylabel is not None:
            axarr[row, col].set_ylabel(ylabel)
        if title is not None:
            axarr[row, col].set_title(title[k])

    if colorbar:
        cbar = f.colorbar(sc)
        if colorbar_labels is not None:
            yticks = cbar.ax.get_yticks()
            ytlabels = colorbar_labels[::len(colors) // (len(yticks) - 1) - 1]
            cbar.set_ticks(yticks)
            cbar.set_ticklabels(ytlabels)

    if savefile is not None:
        plt.savefig(savefile)

    plt.show()


def hist(data, bins=None,
         labels=None, colors=None,
         xlabel=None, ylabel=None,
         title=None, suptitle=None,
         savefile=None, grid=None, figsize=(12, 3)):
    K = len(data)
    if grid is None:
        grid = (1, K)

    f, axarr = plt.subplots(grid[0], grid[1], figsize=figsize,
                            sharey=True)

    if isinstance(axarr, np.ndarray):
        if axarr.ndim == 1:
            axarr = np.array([axarr])
    else:
        axarr = np.array([[axarr]])

    if suptitle is not None:
        plt.suptitle(suptitle)

    for k in range(K):
        row = k // grid[1]
        col = k % grid[1]
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
                axarr[row, col].hist(data[k][i], **kwargs)
        else:
            axarr[row, col].hist(data[k])
        if xlabel is not None:
            axarr[row, col].set_xlabel(xlabel)
        if k == 0:
            axarr[row, col].legend()
            if ylabel is not None:
                axarr[row, col].set_ylabel(ylabel)
        if title is not None:
            axarr[row, col].set_title(title[k])

    if savefile is not None:
        plt.savefig(savefile)

    plt.show()


def multiPredRange(key, actual, predicted,
                   xlabel, ylabel,
                   savefile=None, figsize=(12, 3)):
    f, axarr = plt.subplots(1, len(key), figsize=figsize, sharey=True)
    for (i, k) in enumerate(key):
        idx = np.argsort(actual[k])
        axarr[i].plot(actual[k][idx], label='Actual',
                      alpha=0.7, color='red')
        axarr[i].plot(np.mean(predicted[k], axis=1)[idx], label='Predicted',
                      linestyle='', marker='o', markersize=1)
        axarr[i].fill_between(range(len(idx)),
                              np.min(predicted[k], axis=1)[idx],
                              np.max(predicted[k], axis=1)[idx],
                              alpha=0.5, edgecolor=None)
        axarr[i].set_title(k)
        if xlabel is not None:
            axarr[i].set_xlabel(xlabel)
        if i == 0:
            axarr[i].legend()
            if ylabel is not None:
                axarr[i].set_ylabel(ylabel)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
