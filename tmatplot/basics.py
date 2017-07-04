import numpy as np
import matplotlib.pyplot as plt


def histAll(data, title, grid=(3, 5),
            savefile=None, figsize=(10, 4)):
    f, ax = plt.subplots(grid[0], grid[1], figsize=figsize)
    for i in range(data.shape[1]):
        row = i % grid[0]
        col = i / grid[0]
        ax[row, col].hist(data[~np.isnan(data[:, i]), i], 100,
                          facecolor='#c5c5c5', edgecolor='None')
        ax[row, col].set_title(title[i])
    for i in ax:
        for j in i:
            plt.setp(j.get_xticklabels(), visible=False)
            plt.setp(j.get_yticklabels(), visible=False)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    plt.close()


def corr(data, ylabel=None,
         savefile=None):
    plt.figure()
    plt.imshow(np.corrcoef(data[np.sum(np.isnan(data), axis=1) == 0].T),
               interpolation='nearest')
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    if ylabel is None:
        plt.setp(plt.gca().get_yticklabels(), visible=False)
    else:
        plt.gca().set_yticklabels(ylabel)
    plt.colorbar()
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    plt.close()


def dualBar(data1, data2,
            title=None, xlabel=None,
            ylabel1=None, ylabel2=None, width=3,
            savefile=None, figsize=(8, 3)):
    plt.figure(figsize=figsize)
    ax1 = plt.gca()
    ax2 = plt.gca().twinx()

    K = data1.shape[0]
    ax1.bar(width*np.arange(K)-width*0.2, np.mean(data1, axis=1), width*0.35,
            yerr=np.std(data1, axis=1), color='#999999')
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
             savefile=None, figsize=(8, 2)):
    f, axarr = plt.subplots(1, len(data), figsize=figsize)
    if len(data) == 1:
        axarr = [axarr]
    for (k, d) in enumerate(data):
        axarr[k].bar(range(len(d)), d)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()


def scatter(x, y, xlabel=None, ylabel=None,
            title=None, suptitle=None,
            identityline=False, markersize=1,
            colorbar=False, colorbar_labels=None,
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
        row = k / grid[1]
        col = k % grid[1]
        if identityline:
            axarr[row, col].plot([-5, 5], [-5, 5], 'k--', color='gray')
        sc = axarr[row, col].scatter(x[k], y[k], c=colors,
                                     cmap='rainbow', s=markersize)
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


def multiHist(key, actual, predicted, bins,
              xlabel, ylabel,
              savefile=None, figsize=(12, 3)):
    f, axarr = plt.subplots(1, len(key), figsize=figsize, sharey=True)
    for (i, k) in enumerate(key):
        axarr[i].hist(actual[k], bins=bins,
                      label='Actual', alpha=0.2, color='red')
        axarr[i].hist(predicted[k], bins=bins,
                      label='Predicted', alpha=0.5)
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
