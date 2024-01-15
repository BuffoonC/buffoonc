# coding=UTF-8
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import max, sum
from scipy.spatial.distance import pdist, squareform


def min_max_normalize_2d_array(arr):
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    normalized_array = (arr - min_vals) / (max_vals - min_vals)
    return normalized_array


def draw_cluster(datas, labs, num, dic_colors=None, dic_markers=None, name=None):
    plt.cla()
    for k in range(num):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k], marker=dic_markers[k])
        # plt.scatter(datas[centers[k], 0], datas[centers[k], 1], facecolor='r', edgecolor="k", marker="*", s=100., alpha=1)
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
def draw_cluster(datas, labs, num, dic_colors=None, dic_markers=None, name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 创建 3D 子图

    for k in range(num):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        ax.scatter(sub_datas[:, 0], sub_datas[:, 1], sub_datas[:, 2], s=16., color=dic_colors[k], marker=dic_markers[k])
        # ax.scatter(datas[centers[k], 0], datas[centers[k], 1], datas[centers[k], 2],
        #            facecolor='r', edgecolor="k", marker="*", s=100., alpha=1)
    ax.tick_params(direction='in')
    ax.set_xlabel("x", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    ax.set_ylabel("y", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    ax.set_zlabel("z", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.grid(False)
    plt.savefig(name, dpi=600)
    plt.show()
'''

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


def find_centers(ti, three_stand):
    centers = np.argsort(-ti)
    counter = 0
    for counter in range(len(ti)):
        if ti[centers[counter]] < three_stand:
            break
    return centers[:counter]


def cluster_PD(rho, centers, nearest_higher_neiber):
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
            labs[index] = labs[int(nearest_higher_neiber[index])]
    return labs


def calculate_adjacent_edges_matrix(data, labels, dc):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    adjacent_edges_matrix = np.zeros((num_clusters, num_clusters), dtype=int)
    for cluster_i in range(num_clusters):
        for cluster_j in range(cluster_i + 1, num_clusters):
            adjacent_edges_count = count_adjacent_edges_between_clusters(data, labels, dc, cluster_i, cluster_j)
            adjacent_edges_matrix[cluster_i, cluster_j] = adjacent_edges_count
            adjacent_edges_matrix[cluster_j, cluster_i] = adjacent_edges_count
    return adjacent_edges_matrix


def count_adjacent_edges_between_clusters(data, labels, dc, cluster_i, cluster_j):
    adjacent_edges_count = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            label_i = labels[i]
            label_j = labels[j]
            if (label_i == cluster_i and label_j == cluster_j or
                    label_i == cluster_j and label_j == cluster_i) and np.linalg.norm(data[i] - data[j]) <= dc:
                adjacent_edges_count += 1
    return adjacent_edges_count


def calculate_cluster_sizes(labels):
    max_label = max(labels)
    cluster_sizes = [0] * (max_label + 1)
    for label in labels:
        cluster_sizes[label] += 1
    return cluster_sizes


def count_adjacent_edges(data, labels, dc):
    num_clusters = len(set(labels))
    adjacent_edges_count = {i: 0 for i in range(num_clusters)}
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if labels[i] == labels[j] and np.linalg.norm(data[i] - data[j]) <= dc:
                adjacent_edges_count[labels[i]] += 1
                adjacent_edges_count[labels[j]] += 1
    return adjacent_edges_count


def calculate_cluster_similarity(adjacent_edges_matrix, cluster_sizes):
    num_clusters = len(cluster_sizes)
    similarity_matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            common_edges = adjacent_edges_matrix[i, j]
            similarity_value = common_edges / (cluster_sizes[i] + cluster_sizes[j])
            similarity_matrix[i, j] = similarity_value
            similarity_matrix[j, i] = similarity_value
    return similarity_matrix


def merge_clusters(similarity_matrix, threshold, original_labels):
    num_clusters = len(similarity_matrix)
    merged_labels = np.arange(num_clusters)
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            if similarity_matrix[i, j] > threshold:
                merged_labels[merged_labels == merged_labels[j]] = merged_labels[i]
    unique_labels = np.unique(merged_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    merged_labels = np.array([label_mapping[label] for label in merged_labels])
    updated_labels = merged_labels[original_labels]
    num_merged_clusters = len(unique_labels)
    return updated_labels, num_merged_clusters


if __name__ == "__main__":
    dic_markers = {0: 'o', 1: 's', 2: '^', 3: 'v', 4: '<', 5: '>', 6: 'x', 7: '+', 8: 'd', 9: '*', 10: 'p', 11: 'h',
                   12: 'H', 13: '8', 14: 'o', 15: 's', 16: '^', 17: 'D', 18: 'P', 19: 'X', 20: 'o', 21: 's', 22: '^',
                   23: 'v', 24: '<', 25: '>', 26: 'x', 27: '+', 28: 'd', 29: '*', 30: 'p', 31: 'h',
                   32: 'H', 33: '8', 34: 'o', 35: 's', 36: '^', 37: 'D', 38: 'P', 39: 'X'}
    num_colors = 50
    tab10_colors = plt.cm.tab10.colors
    dic_colors = {i: tab10_colors[i % 10] for i in range(num_colors)}
    '''
    # 加载D31数据集，threshold = 1, 标准差 = 3，0.9383/0.9583/0.9403
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\D31.txt', sep=',', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\D31.txt', sep=',', engine='python', usecols=[2],header=None)).flatten()
    file_name = 'D31'
    '''
    '''
    # 加载jain数据集, threshold = 0.004, 标准差 = 2， 1/1/1
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\jain.txt', sep=',', engine='python', usecols=[1, 2],header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\jain.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()
    file_name = 'jain'
    '''
    '''
    # 加载ED_Hexagon数据集, threshold = 0.03 , 标准差 = 3, 1/1/1
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\ED_Hexagon.txt', sep=',', engine='python', usecols=[0, 1], header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\ED_Hexagon.txt', sep=',', engine='python', usecols=[2], header=None)).flatten()
    file_name = 'ED_Hexagon'
    '''

    # 加载donutcurves数据集，threshold = 0.01 , 标准差 = 0.8, 1/1/1
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\donutcurves.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\donutcurves_cl.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()
    file_name = 'donutcurves'

    '''
    # 加载banana数据集，threshold = 0.03 , 标准差 = 3, 1/1/1
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\banana.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\banana_cl.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()
    file_name = 'banana'
    '''
    '''
    # 加载chainlink数据集, threshold = 0.01 , 标准差 = 2, 1/1/1
    selected_columns = list(range(0, 3))
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\chainlink.txt', sep='\t', engine='python', usecols=selected_columns, header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\chainlink_cl.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()
    file_name = 'chainlink'
    # normalized_data = min_max_normalize_2d_array(dataset)
    normalized_data = dataset
    '''
    normalized_data = min_max_normalize_2d_array(dataset)
    dists = squareform(pdist(normalized_data))
    dc = CutoffDistance(dists)
    print(f'截断距离dc: {dc:.4f}')
    rho = get_density(dists, dc)
    higher_deltas, nearest_higher_neiber = get_maxdeltas(dists, rho)
    ti = get_ti(higher_deltas, rho)
    ti_avg = ti_average(ti)
    ti_standd = standdeviation(ti)
    ti_zscore = get_z_score(ti)
    threshold = 0.01
    confi_h = (ti_average(ti_zscore) + 0.8 * get_variance(ti_zscore)) * ti_standd + ti_avg
    centers = find_centers(ti, confi_h)
    labs = cluster_PD(rho, centers, nearest_higher_neiber)
    result_cluster_sizes = calculate_cluster_sizes(labs)
    result_adjacent_edges_matrix = calculate_adjacent_edges_matrix(normalized_data, labs, dc)
    result_cluster_similarity = calculate_cluster_similarity(result_adjacent_edges_matrix, result_cluster_sizes)
    updated_labels, num_merged_clusters = merge_clusters(result_cluster_similarity, threshold, labs)
    draw_cluster(normalized_data, updated_labels, num_merged_clusters, dic_colors, dic_markers, name=file_name + "_cluster_GDPC.svg")
    ari = adjusted_rand_score(true_labels, updated_labels)
    print(f'调整兰德指数（ARI）: {ari:.4f}')
    nmi = normalized_mutual_info_score(true_labels, updated_labels)
    print(f'归一化互信息（NMI）: {nmi:.4f}')
    fmi = fowlkes_mallows_score(true_labels, updated_labels)
    print(f"Fowlkes-Mallows Index（FMI）: {fmi:.4f}")
