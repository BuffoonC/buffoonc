import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

UNCLASSIFIED = 0
NOISE = -1


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
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists


def find_points_in_eps(point_id, eps, dists):
    index = (dists[point_id] <= eps)
    return np.where(index == True)[0].tolist()


def expand_cluster(dists, labs, cluster_id, seeds, eps, min_points):
    i = 0
    while i < len(seeds):
        Pn = seeds[i]
        if labs[Pn] == NOISE:
            labs[Pn] = cluster_id
        elif labs[Pn] == UNCLASSIFIED:
            labs[Pn] = cluster_id
            new_seeds = find_points_in_eps(Pn, eps, dists)
            if len(new_seeds) >= min_points:
                seeds = seeds + new_seeds
        i = i + 1


def dbscan(datas, eps, min_points):
    dists = getDistanceMatrix(datas)
    n_points = datas.shape[0]
    labs = [UNCLASSIFIED] * n_points
    cluster_id = 0
    for point_id in range(0, n_points):
        if not (labs[point_id] == UNCLASSIFIED):
            continue
        seeds = find_points_in_eps(point_id, eps, dists)
        if len(seeds) < min_points:
            labs[point_id] = NOISE
        else:
            cluster_id = cluster_id + 1
            labs[point_id] = cluster_id
            expand_cluster(dists, labs, cluster_id, seeds, eps, min_points)
    return labs, cluster_id


# 绘图
def draw_cluster(datas, labs, n_cluster, dic_colors, name=None):
    plt.cla()
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, n_cluster)]
    for i, lab in enumerate(labs):
        if lab == NOISE:
            plt.scatter(datas[i, 0], datas[i, 1], s=16., color=(0, 0, 0))
        else:
            plt.scatter(datas[i, 0], datas[i, 1], s=16., color=dic_colors[lab])
    plt.savefig(name, dpi = 600)
    plt.show()


if __name__ == "__main__":
    # 31种颜色
    num_colors = 50
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

    # # 加载Glass数据集
    # selected_columns = list(range(1, 10))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Glass.txt', sep=',', engine='python',usecols=selected_columns, header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Glass.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()

    # # 加载Ecoli数据集
    # selected_columns = list(range(1, 8))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Ecoli.txt', sep=',', engine='python', usecols=selected_columns,header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Ecoli.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()

    # # 加载Blood数据集
    # selected_columns = list(range(1, 5))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Iris.txt', sep=',', engine='python', usecols=selected_columns, header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Iris.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()

    # # 加载Wine数据集
    # selected_columns = list(range(1, 14))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Wine.txt', sep=',', engine='python',usecols=selected_columns, header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\Wine.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()
    normalized_data = min_max_normalize_2d_array(dataset)
    max_ari = -1
    best_metrics = {}
    for i in range(1, 21):
        for j in [round(x * 0.01, 2) for x in range(1, 101)]:
            labels, cluster_id = dbscan(normalized_data, eps=j, min_points=i)
            ari = adjusted_rand_score(true_labels, labels)
            nmi = normalized_mutual_info_score(true_labels, labels)
            fmi = fowlkes_mallows_score(true_labels, labels)
            print('=========================================')
            print(f'ARI: {ari:.4f}')
            print(f'NMI: {nmi:.4f}')
            print(f'FMI: {fmi:.4f}')
            print(f'min_points: {i:.4f}')
            print(f'eps: {j:.4f}')
            if ari > max_ari:
                max_ari = ari
                best_metrics = {
                    'ARI': ari,
                    'NMI': nmi,
                    'FMI': fmi,
                    'min_points': i,
                    'eps': j
                }
    # 打印ARI最大时的相关指标和参数
    print('=========================================')
    print(f'最大ARI时的相关指标和参数:')
    print(f'ARI: {best_metrics["ARI"]:.4f}')
    print(f'NMI: {best_metrics["NMI"]:.4f}')
    print(f'FMI: {best_metrics["FMI"]:.4f}')
    print(f'min_points: {best_metrics["min_points"]:.4f}')
    print(f'eps: {best_metrics["eps"]:.4f}')

