import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score


def getRho(n, distMatrix,k):
    rho = np.zeros(n,dtype=float)
    for i in range(n):
        distKnn = np.argsort(distMatrix[i])
        distKnn_unique = np.unique(distMatrix[i, distKnn])
        for j in range(1, min(k + 1, len(distKnn_unique))):
            rho[i] = rho[i] + math.exp(-distKnn_unique[j] * distKnn_unique[j])
    return rho


def hexindianshibie(n, k, distMatrix, rho):
    maxrhoindex = np.zeros(n, dtype=int)
    for i in range(n):
        maxrho = rho[i]
        maxrhoindex[i] = i
        distKnn = np.argsort(distMatrix[i])
        for j in range(1, k+1):
            if j < len(distKnn) and rho[distKnn[j]] > maxrho:
                maxrho = rho[distKnn[j]]
                maxrhoindex[i] = distKnn[j]
    return maxrhoindex


def delete_repeat_hexindian(maxrhoindex):
    finalmaxrhoindex=np.unique(maxrhoindex)
    return finalmaxrhoindex


def hexindianjicheng(finalmaxrhoindex,n,k):
    ncore = finalmaxrhoindex.shape[0]
    a = 2*((ncore/n)+0.1)
    ka = int(a*k)
    union_Set = []
    for i in range(ncore-1):
        distKnni = np.argsort(distMatrix[finalmaxrhoindex[i]]).tolist()
        distKnni = distKnni[1:ka+1]
        for j in range(i+1,ncore):
            distKnnj = np.argsort(distMatrix[finalmaxrhoindex[j]]).tolist()
            distKnnj = distKnnj[1:ka+1]
            if len(set(distKnni).intersection(set(distKnnj))) != 0:
                union_Set.append([finalmaxrhoindex[i], finalmaxrhoindex[j]])
    return union_Set


def make_set(count):
    parent = []
    for i in range(count):
        parent.append(i)
    return parent


def Find_Root(x, parent):
    if x != parent[x]:
        parent[x] = Find_Root(parent[x],parent)
    return parent[x]


def Union(x, y, parent):
    xroot = Find_Root(x, parent)
    yroot = Find_Root(y, parent)
    if xroot == yroot:
        return
    elif yroot > xroot:
        parent[yroot] = xroot
    else:
        parent[xroot] = yroot


def final_Parent(union_Set, count):
    parent = make_set(count)
    for i in range(len(union_Set)):
        Union(union_Set[i][0], union_Set[i][1],parent)
    for i in range(count):
        Find_Root(i,parent)
    return parent


def min_max_normalize_2d_array(arr):
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    normalized_array = (arr - min_vals) / (max_vals - min_vals)
    return normalized_array


'''
def draw_cluster(datas, labs, num, dic_colors, dic_markers, name="0_cluster.svg"):
    plt.cla()
    for k in range(num):
        sub_index = np.flatnonzero(labs == k)
        sub_datas = datas[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k], marker=dic_markers[k])
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


def draw_cluster(datas, labs, num, dic_colors, dic_markers, name="0_cluster.svg"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k in range(num):
        sub_index = np.flatnonzero(labs == k)
        sub_datas = datas[sub_index]
        ax.scatter(sub_datas[:, 0], sub_datas[:, 1], sub_datas[:, 2], s=16., color=dic_colors[k], marker=dic_markers[k])
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("x", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    ax.set_ylabel("y", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    ax.set_zlabel("z", fontsize=11, fontdict={'family': 'Times New Roman', 'style': 'italic'})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.grid(False)
    plt.savefig(name, dpi=600)
    plt.show()


if __name__ == '__main__':
    dic_markers = {0: 'o', 1: 's', 2: '^', 3: 'v', 4: '<', 5: '>', 6: 'x', 7: '+', 8: 'd', 9: '*', 10: 'p', 11: 'h',
                   12: 'H', 13: '8', 14: 'o', 15: 's', 16: '^', 17: 'D', 18: 'P', 19: 'X', 20: 'o', 21: 's', 22: '^',
                   23: 'v', 24: '<', 25: '>', 26: 'x', 27: '+', 28: 'd', 29: '*', 30: 'p', 31: 'h',
                   32: 'H', 33: '8', 34: 'o', 35: 's', 36: '^', 37: 'D', 38: 'P', 39: 'X'}
    num_colors = 1000
    tab10_colors = plt.cm.tab10.colors
    dic_colors = {i: tab10_colors[i % 10] for i in range(num_colors)}
    '''
    # 加载D31数据集，gama = 0.012, 0.9497/0.9653/0.9513
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\D31.txt', sep=',', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\D31.txt', sep=',', engine='python', usecols=[2],header=None)).flatten()
    file_name = 'D31'
    '''
    '''
    # 加载jain数据集, gama = 0.45, 0.7305/0.6092/0.8912
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\jain.txt', sep=',', engine='python', usecols=[1, 2], header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\jain.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()
    file_name = 'jain'
    '''
    '''
    # 加载ED_Hexagon数据集，gama = 0.08, 1/1/1
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\ED_Hexagon.txt', sep=',', engine='python', usecols=[0, 1], header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\ED_Hexagon.txt', sep=',', engine='python', usecols=[2], header=None)).flatten()
    file_name = 'ED_Hexagon'
    '''
    '''
    # 加载donutcurves数据集，gama = 0.13, 0.7137/0.8571/0.8160
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\donutcurves.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\donutcurves_cl.txt', sep=',', engine='python', usecols=[0],header=None)).flatten()
    file_name = 'donutcurves'
    '''
    '''
    # 加载banana数据集，gama = 0.14, 0.3696/0.4156/0.6981
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\banana.txt', sep='\t', engine='python', usecols=[0, 1], header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\banana_cl.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()
    file_name = 'banana'
    '''
    # 加载chainlink数据集, gama = 0.12, 0.2892/0.5153/0.5376
    selected_columns = list(range(0, 3))
    dataset = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\chainlink.txt', sep='\t', engine='python', usecols=selected_columns, header=None))
    true_labels = np.array(pd.read_table(r'D:\毕业设计（认认真真版）\数据集\chainlink_cl.txt', sep=',', engine='python', usecols=[0], header=None)).flatten()
    file_name = 'chainlink'
    # normalized_data = min_max_normalize_2d_array(dataset)
    normalized_data = dataset
    '''
    # 加载t4数据集, gama = 0.15, 0.3913/0.5478/0.4953
    dataset = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\t4.txt', sep='\t', engine='python', usecols=[0, 1],header=None))
    true_labels = np.array(pd.read_table(r'D:\pythonProject\artificial_dataset\t4_cl.txt', sep='\t', engine='python', usecols=[0],header=None)).flatten()
    file_name = 't4'
    '''
    # normalized_data = min_max_normalize_2d_array(dataset)
    x = dataset[:, 0]
    y = dataset[:, 1]
    n = x.shape[0]
    distList = pdist(dataset, metric='euclidean')
    distMatrix = squareform(distList)
    # for i in range(10, 51):
    gama = 0.12
    k = int(gama*n)
    rho = getRho(n, distMatrix, k)
    maxrhoindex = hexindianshibie(n, k, distMatrix, rho)
    finalmaxrhoindex = delete_repeat_hexindian(maxrhoindex)
    union_Set = hexindianjicheng(finalmaxrhoindex, n, k)
    clusterSet = OrderedDict()
    for i in range(len(finalmaxrhoindex)):
        clusterSet[finalmaxrhoindex[i]] = i
    for i in range(len(union_Set)):
        union_Set[i][0] = clusterSet[union_Set[i][0]]
        union_Set[i][1] = clusterSet[union_Set[i][1]]
    final_parent = final_Parent(union_Set, len(finalmaxrhoindex))
    aset = set(final_parent)
    for i in range(len(finalmaxrhoindex)):
        final_parent[i] = finalmaxrhoindex[final_parent[i]]
    cluster = []
    for i in range(n):
        cluster.append(-1)
    for i in range(len(finalmaxrhoindex)):
        cluster[finalmaxrhoindex[i]] = final_parent[i]
    for i in range(n):
        cluster[i] = cluster[maxrhoindex[i]]
    unique_elements, labels = np.unique(cluster, return_inverse=True)
    draw_cluster(normalized_data, labels, len(unique_elements), dic_colors, dic_markers, name=file_name + "_cluster_ICKDC.svg")
    ari = adjusted_rand_score(true_labels, cluster)
    print(f'调整兰德指数（ARI）: {ari:.4f}')
    nmi = normalized_mutual_info_score(true_labels, cluster)
    print(f'归一化互信息（NMI）: {nmi:.4f}')
    fmi = fowlkes_mallows_score(true_labels, cluster)
    print(f"Fowlkes-Mallows Index（FMI）: {fmi:.4f}")
