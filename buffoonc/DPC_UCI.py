import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score


def min_max_normalize_2d_array(arr):
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    normalized_array = (arr - min_vals) / (max_vals - min_vals)
    return normalized_array


def getDistanceMatrix(datas):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            dists[i, j] = distance.euclidean(datas[i], datas[j])
    return dists


def select_dc(dists):
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


def get_density(dists, dc, method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


def get_deltas(dists, rho):
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


def find_centers_auto(rho, deltas):
    rho_threshold = (np.min(rho) + np.max(rho)) / 2
    delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
    N = np.shape(rho)[0]
    centers = []
    for i in range(N):
        if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
            centers.append(i)
    return np.array(centers)


def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return
    N = np.shape(rho)[0]
    labs = -1 * np.ones(N).astype(int)
    for i, center in enumerate(centers):
        labs[center] = i
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        if labs[index] == -1:
            labs[index] = labs[int(nearest_neiber[index])]
    return labs


def draw_decision(rho, deltas, name=None):
    plt.cla()
    for i in range(np.shape(dataset)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    plt.savefig(name, dpi=600)
    plt.show()


def draw_cluster(datas, labs, centers, dic_colors=None, name=None):
    plt.cla()
    K = np.shape(centers)[0]

    for k in range(K):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color= dic_colors[k])
        plt.scatter(datas[centers[k], 0], datas[centers[k], 1], facecolor='r', edgecolor="k", marker="*", s=200., alpha=1)
    plt.savefig(name, dpi=600)
    plt.show()


if __name__ == "__main__":
    # 31种颜色
    num_colors = 31
    tab10_colors = plt.cm.tab10.colors
    dic_colors = {i: tab10_colors[i % 10] for i in range(num_colors)}

    # # 加载Liver数据集，参数K=2
    # selected_columns = list(range(1, 7))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\liver.txt', sep=' ', engine='python',usecols=selected_columns, header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\liver.txt', sep=' ', engine='python', usecols=[0],header=None)).flatten()

    # 加载Wpbc数据集，参数K=2
    selected_columns = list(range(1, 34))
    dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\WDBC.txt', sep=',', engine='python', usecols=selected_columns, header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\WDBC.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()

    # # 加载Glass数据集，参数K=6
    # selected_columns = list(range(1, 10))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Glass.txt', sep=',', engine='python',usecols=selected_columns, header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Glass.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()

    # # 加载Ecoli数据集，参数K=8
    # selected_columns = list(range(1, 8))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Ecoli.txt', sep=',', engine='python', usecols=selected_columns,header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Ecoli.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()

    # # 加载Blood数据集，参数K=2
    # selected_columns = list(range(1, 5))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Iris.txt', sep=',', engine='python', usecols=selected_columns, header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Iris.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()

    # # 加载Wine数据集，参数K=3
    # selected_columns = list(range(1, 14))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Wine.txt', sep=',', engine='python', usecols=selected_columns, header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Wine.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()

    normalized_data = min_max_normalize_2d_array(dataset)

    dists = getDistanceMatrix(normalized_data)
    dc = select_dc(dists)
    rho = get_density(dists, dc, method="Gaussion")
    deltas, nearest_neiber = get_deltas(dists, rho)
    centers = find_centers_K(rho, deltas, 2)
    labs = cluster_PD(rho, centers, nearest_neiber)
    ari = adjusted_rand_score(true_labels, labs)
    print(f'调整兰德指数（ARI）: {ari:.4f}')
    nmi = normalized_mutual_info_score(true_labels, labs)
    print(f'归一化互信息（NMI）: {nmi:.4f}')
    fmi = fowlkes_mallows_score(true_labels, labs)
    print(f"Fowlkes-Mallows Index（FMI）: {fmi:.4f}")
