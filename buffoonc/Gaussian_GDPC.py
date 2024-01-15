# coding=UTF-8
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.stats import norm
from numpy import sum
from scipy.spatial.distance import pdist, squareform


def min_max_normalize_2d_array(arr):
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    normalized_array = (arr - min_vals) / (max_vals - min_vals)
    return normalized_array


def CutoffDistance(dists):
    N = np.shape(dists)[0]
    max_dis = np.max(dists)
    min_dis = np.min(dists)
    dc = (max_dis + min_dis) / 2
    while True:
        n_neighs = np.where(dists < dc)[0].shape[0] - N
        rate = n_neighs / (N * (N - 1))
        if rate >= 0.01 and rate <= 0.02:
            break
        if rate < 0.01:
            min_dis = dc
        else:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break
    return dc


def get_density(dists, dc):

    N = dists.shape[0]
    rho = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if dists[i, j] <= dc:
                rho[i] += 1
    return rho



def get_maxdeltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        if i == 0:
            continue
        index_higher_rho = index_rho[:i]
        deltas[index] = np.min(dists[index, index_higher_rho])
        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)
    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


def get_ti(higher_deltas, rho):
    N = np.shape(higher_deltas)[0]
    distdiff = np.zeros(N)
    for i in range(N):
        distdiff[i] = higher_deltas[i]*rho[0]
    return distdiff


def ti_average(ti):
    return sum(ti) / len(ti)


def get_variance(ti):
    average = ti_average(ti)
    return sum([(x - average) ** 2 for x in ti]) / len(ti)


def standdeviation(ti):
    variance = get_variance(ti)
    return math.sqrt(variance)


def get_z_score(data):
    avg = ti_average(data)
    stan = standdeviation(data)
    scores = [(i - avg) / stan for i in data]
    return scores


if __name__ == "__main__":
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\flame2.txt', sep='\t', usecols=[0, 1], engine='python', header=None))
    normalized_data = min_max_normalize_2d_array(dataset)
    dists = squareform(pdist(normalized_data))
    dc = CutoffDistance(dists)
    rho = get_density(dists, dc)
    higher_deltas, nearest_higher_neiber = get_maxdeltas(dists, rho)
    ti = get_ti(higher_deltas, rho)
    ti_zscore = get_z_score(ti)
    mu = ti_average(ti_zscore)
    std = standdeviation(ti_zscore)
    xmin, xmax = mu - 3 * std, mu + 3 * std
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.figure(figsize=(8, 4))
    plt.hist(ti_zscore, bins=240, density=True, alpha=0.6, color='r')
    plt.plot(x, p, 'k', linewidth=2)
    ax = plt.gca()
    ax.tick_params(direction='in')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.xlim(-4, 10)
    plt.ylim(0, 4)
    plt.xlabel("x", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    plt.ylabel("f(x)", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    plt.savefig("gaussian_distribution.svg", dpi=600)
    plt.show()

