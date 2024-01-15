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


'''
def draw_cluster(datas, labs, n_cluster, dic_colors, dic_markers, name=None, NOISE=-1):
    plt.cla()
    for i, lab in enumerate(labs):
        if lab == NOISE:
            plt.scatter(datas[i, 0], datas[i, 1], s=16., color=(0, 0, 0), marker='x')
        else:
            plt.scatter(datas[i, 0], datas[i, 1], s=16., color=dic_colors[lab], marker=dic_markers[lab])
    ax = plt.gca()
    ax.tick_params(direction='in')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(-0.1, 1.2)
    plt.ylim(-0.1, 1.2)
    plt.xlabel("x", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    plt.ylabel("y", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    if name:
        plt.savefig(name, dpi=600)
    plt.show()
'''


def draw_cluster(datas, labs, n_cluster, dic_colors, dic_markers, name=None, NOISE=-1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, lab in enumerate(labs):
        if lab == NOISE:
            ax.scatter(datas[i, 0], datas[i, 1], datas[i, 2], s=16., color=(0, 0, 0), marker='x')
        else:
            ax.scatter(datas[i, 0], datas[i, 1], datas[i, 2], s=16., color=dic_colors[lab], marker=dic_markers[lab])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    ax.zaxis.set_tick_params(direction='in')
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.zaxis.label.set_size(11)
    ax.xaxis.label.set_fontfamily('Times New Roman')
    ax.yaxis.label.set_fontfamily('Times New Roman')
    ax.zaxis.label.set_fontfamily('Times New Roman')
    ax.xaxis.label.set_fontstyle('italic')
    ax.yaxis.label.set_fontstyle('italic')
    ax.zaxis.label.set_fontstyle('italic')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.grid(False)
    ax.tick_params(direction='in')
    if name:
        plt.savefig(name, dpi=600)
    plt.show()


if __name__ == "__main__":
    dic_markers = {0: 'o', 1: 's', 2: '^', 3: 'v', 4: '<', 5: '>', 6: 'x', 7: '+', 8: 'd', 9: '*', 10: 'p', 11: 'h',
                   12: 'H', 13: '8', 14: 'o', 15: 's', 16: '^', 17: 'D', 18: 'P', 19: 'X', 20: 'o', 21: 's', 22: '^',
                   23: 'v', 24: '<', 25: '>', 26: 'x', 27: '+', 28: 'd', 29: '*', 30: 'p', 31: 'h',
                   32: 'H', 33: '8', 34: 'o', 35: 's', 36: '^', 37: 'D', 38: 'P', 39: 'X'}
    num_colors = 50
    tab10_colors = plt.cm.tab10.colors
    dic_colors = {i: tab10_colors[i % 10] for i in range(num_colors)}
    '''
    # 加载D31数据集，eps = 0.03, min_points = 30, 0.5400/0.8378/0.5716
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\D31.txt', sep=',', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\D31.txt', sep=',', engine='python', usecols=[2],header=None)).flatten()
    file_name = 'D31'
    '''
    '''
    # 加载jain数据集, eps = 0.08, min_points = 5, 0.9731/0.9178/0.9895
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\jain.txt', sep=',', engine='python', usecols=[1, 2], header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\jain.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()
    file_name = 'jain'
    '''
    '''
    # 加载ED_Hexagon数据集，eps = 0.08, min_points = 5, 1/1/1
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\ED_Hexagon.txt', sep=',', engine='python', usecols=[0, 1], header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\ED_Hexagon.txt', sep=',', engine='python', usecols=[2], header=None)).flatten()
    file_name = 'ED_Hexagon'
    '''
    '''
    # 加载donutcurves数据集，eps = 0.04, min_points = 5, 1/1/1
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\donutcurves.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\donutcurves_cl.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()
    file_name = 'donutcurves'
    '''
    '''
    # 加载banana数据集，eps = 0.08, min_points = 30, 1/1/1
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\banana.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\banana_cl.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()
    file_name = 'banana'
    '''
    # 加载chainlink数据集, eps = 0.12, min_points = 5, 1/1/1
    selected_columns = list(range(0, 3))
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\chainlink.txt', sep='\t', engine='python', usecols=selected_columns, header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\chainlink_cl.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()
    file_name = 'chainlink'
    # normalized_data = min_max_normalize_2d_array(dataset)
    normalized_data = dataset
    '''
    # 加载t4数据集, eps = 0.03, min_points = 30, 0.9051/0.8982/0.9238
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\t4.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\t4_cl.txt', sep='\t', engine='python', usecols=[0],header=None)).flatten()
    file_name = 't4'
    '''
    # normalized_data = min_max_normalize_2d_array(dataset)
    eps = 0.12
    min_points = 5
    labels, cluster_id = dbscan(normalized_data, eps=eps, min_points=min_points)
    draw_cluster(normalized_data, labels, cluster_id, dic_colors, dic_markers, name=file_name+"_cluster_DBSCAN.svg")
    ari = adjusted_rand_score(true_labels, labels)
    print(f'调整兰德指数（ARI）: {ari:.4f}')
    nmi = normalized_mutual_info_score(true_labels, labels)
    print(f'归一化互信息（NMI）: {nmi:.4f}')
    fmi = fowlkes_mallows_score(true_labels, labels)
    print(f"Fowlkes-Mallows Index（FMI）: {fmi:.4f}")

