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


def draw_cluster(datas, labs, num, dic_colors=None, dic_markers=None, name=None):
    plt.cla()
    for k in range(len(num)):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k], marker=dic_markers[k])
        plt.scatter(datas[centers[k], 0], datas[centers[k], 1], facecolor='r', edgecolor="k", marker="*", s=100., alpha=1)
    ax = plt.gca()
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(-0.1, 1.2)
    plt.ylim(-0.1, 1.2)
    plt.xlabel("x", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    plt.ylabel("y", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    plt.savefig(name, dpi=600)
    plt.show()
'''
def draw_cluster(datas, labs, centers, dic_colors=None, dic_markers=None, name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.cla()
    for k in range(len(centers)):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        ax.scatter(sub_datas[:, 0], sub_datas[:, 1], sub_datas[:, 2], s=16., color=dic_colors[k], marker=dic_markers[k])
        ax.scatter(datas[centers[k], 0], datas[centers[k], 1], datas[centers[k], 2], facecolor='r', edgecolor="k", marker="*", s=100., alpha=1)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.grid(False)
    ax.tick_params(direction='in')
    ax.set_xlabel("x", fontsize=20, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    ax.set_ylabel("y", fontsize=20, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    ax.set_zlabel("z", fontsize=20, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    if name:
        plt.savefig(name, dpi=600)
    plt.show()
'''

if __name__ == "__main__":
    dic_markers = {0: 'o', 1: 's', 2: '^', 3: 'v', 4: '<', 5: '>', 6: 'x', 7: '+', 8: 'd', 9: '*', 10: 'p', 11: 'h',
                   12: 'H', 13: '8', 14: 'o', 15: 's', 16: '^', 17: 'D', 18: 'P', 19: 'X', 20: 'o', 21: 's', 22: '^', 23: 'v', 24: '<', 25: '>', 26: 'x', 27: '+', 28: 'd', 29: '*', 30: 'p', 31: 'h',
                   32: 'H', 33: '8', 34: 'o', 35: 's', 36: '^', 37: 'D', 38: 'P', 39: 'X'}
    num_colors = 31
    tab10_colors = plt.cm.tab10.colors
    dic_colors = {i: tab10_colors[i % 10] for i in range(num_colors)}
    '''
    # 加载D31数据集，0.9358/0.9573/0.9378
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\D31.txt', sep=',', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\D31.txt', sep=',', engine='python', usecols=[2],header=None)).flatten()
    file_name = 'D31'
    '''
    '''
    # 加载jain数据集, 0.7055/0.6447/0.8779
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\jain.txt', sep=',', engine='python', usecols=[1, 2], header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\jain.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()
    file_name = 'jain'
    '''
    '''
    # 加载ED_Hexagon数据集，-0.0901/0.1116/0.6319
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\ED_Hexagon.txt', sep=',', engine='python', usecols=[0, 1], header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\ED_Hexagon.txt', sep=',', engine='python', usecols=[2], header=None)).flatten()
    file_name = 'ED_Hexagon'
    '''
    '''
    # 加载donutcurves数据集，0.7593/0.8478/0.8243
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\donutcurves.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\donutcurves_cl.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()
    file_name = 'donutcurves'
    '''
    '''
    # 加载banana数据集，0.0469/0.0331/0.5303
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\banana.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\banana_cl.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()
    file_name = 'banana'
    '''
    '''
    # 加载chainlink数据集, 0.3312/0.4014/0.6953
    selected_columns = list(range(0, 3))
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\chainlink.txt', sep='\t', engine='python', usecols=selected_columns, header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\chainlink_cl.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()
    file_name = 'chainlink'
    # normalized_data = min_max_normalize_2d_array(dataset)
    normalized_data = dataset
    '''
    # 加载t4数据集,0.6055/0.7349/0.6804
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\t4.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\t4_cl.txt', sep='\t', engine='python', usecols=[0],header=None)).flatten()
    file_name = 't4'
    normalized_data = min_max_normalize_2d_array(dataset)
    dists = getDistanceMatrix(normalized_data)
    dc = select_dc(dists)
    print(f'截断距离dc: {dc:.4f}')
    rho = get_density(dists, dc, method="Gaussion")
    deltas, nearest_neiber = get_deltas(dists, rho)
    # draw_decision(rho, deltas, name=file_name + "_decision_DPC.svg")
    centers = find_centers_K(rho, deltas, 6)
    labs = cluster_PD(rho, centers, nearest_neiber)
    draw_cluster(normalized_data, labs, centers, dic_colors, dic_markers, name=file_name + "_cluster_DPC.svg")
    ari = adjusted_rand_score(true_labels, labs)
    print(f'调整兰德指数（ARI）: {ari:.4f}')
    nmi = normalized_mutual_info_score(true_labels, labs)
    print(f'归一化互信息（NMI）: {nmi:.4f}')
    fmi = fowlkes_mallows_score(true_labels, labs)
    print(f"Fowlkes-Mallows Index（FMI）: {fmi:.4f}")
