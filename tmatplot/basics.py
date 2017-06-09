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


def multiScatter(actual, predicted, title,
                 savefile=None, figsize=(8, 3)):
    import matplotlib.cm as cm
    K = len(actual)
    colors = cm.rainbow(np.linspace(0, 1, len(actual[0])))
    f, axarr = plt.subplots(1, K, figsize=figsize)
    for k in range(K):
        axarr[k].scatter(actual[k], predicted[k], color=colors, s=1)
        axarr[k].plot([-5, 5], [-5, 5], 'k--', color='gray')
        axarr[k].set_xlabel('Actual')
        axarr[k].set_ylabel('Predicted')
        axarr[k].set_title(title[k])
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
