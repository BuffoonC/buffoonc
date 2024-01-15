from random import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score


def getRho(n,distMatrix,k):
    rho = np.zeros(n,dtype=float)
    for i in range(n):
        distKnn=np.argsort(distMatrix[i])
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
    ncore=finalmaxrhoindex.shape[0]
    a=2*((ncore/n)+0.1)
    ka=int(a*k)
    union_Set = []
    for i in range(ncore-1):
        distKnni = np.argsort(distMatrix[finalmaxrhoindex[i]]).tolist()
        distKnni = distKnni[1:ka+1]
        for j in range(i+1,ncore):
            distKnnj = np.argsort(distMatrix[finalmaxrhoindex[j]]).tolist()
            distKnnj=distKnnj[1:ka+1]
            if len(set(distKnni).intersection(set(distKnnj)))!=0:
                union_Set.append([finalmaxrhoindex[i], finalmaxrhoindex[j]])
    return union_Set


def make_set(count):
    parent=[]
    for i in range(count):
        parent.append(i)
    return parent

def Find_Root(x,parent):
    if x != parent[x]:
        parent[x] = Find_Root(parent[x],parent)
    return parent[x]

def Union(x,y,parent):
    xroot = Find_Root(x,parent)
    yroot = Find_Root(y,parent)
    if xroot == yroot:
        return
    elif yroot > xroot:
        parent[yroot] = xroot
    else:
        parent[xroot] = yroot

def final_Parent(union_Set,count):
    parent = make_set(count)
    for i in range(len(union_Set)):
        Union(union_Set[i][0],union_Set[i][1],parent)
    for i in range(count):
        Find_Root(i,parent)
    return parent


def min_max_normalize_2d_array(arr):
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    normalized_array = (arr - min_vals) / (max_vals - min_vals)
    return normalized_array


def draw_cluster(datas, labs, num, dic_colors, name="0_cluster.jpg"):
    plt.cla()
    for k in range(num):
        sub_index = np.flatnonzero(labs == k)
        sub_datas = datas[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color= dic_colors[k])
    plt.savefig(name, dpi=600)
    plt.show()

if __name__ == '__main__':
    # 31种颜色
    num_colors = 1000
    tab10_colors = plt.cm.tab10.colors
    dic_colors = {i: tab10_colors[i % 10] for i in range(num_colors)}
    # # 加载Liver数据集，参数K=2
    # selected_columns = list(range(1, 7))
    # dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\liver.txt', sep=' ', engine='python',usecols=selected_columns, header=None))
    # true_labels = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\liver.txt', sep=' ', engine='python', usecols=[0],header=None)).flatten()

    # 加载Wpbc数据集，参数K=2
    selected_columns = list(range(1, 34))
    dataset = np.array(pd.read_table(r'D:\pythonProject\UCI\UCI数据集txt格式\txt\WDBC.txt', sep=',', engine='python', usecols = selected_columns, header=None))
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
    x = dataset[:, 0]
    y = dataset[:, 1]
    n = x.shape[0]
    distList = pdist(dataset, metric='euclidean')
    distMatrix = squareform(distList)
    max_ari = -1
    best_metrics = {}
    for j in [round(x * 0.01, 2) for x in range(1, 201) if x * 0.01 <= 2]:
        gama = j
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
        final_parent = final_Parent(union_Set,len(finalmaxrhoindex))
        aset = set(final_parent)
        for i in range(len(finalmaxrhoindex)):
            final_parent[i] = finalmaxrhoindex[final_parent[i]]
        cluster=[]
        for i in range(n):
            cluster.append(-1)
        for i in range(len(finalmaxrhoindex)):
            cluster[finalmaxrhoindex[i]] = final_parent[i]
        for i in range(n):
            cluster[i] = cluster[maxrhoindex[i]]
        unique_elements, labels = np.unique(cluster, return_inverse=True)
        ari = adjusted_rand_score(true_labels, cluster)
        nmi = normalized_mutual_info_score(true_labels, cluster)
        fmi = fowlkes_mallows_score(true_labels, cluster)
        print('=================================')
        print(f'调整兰德指数（ARI）: {ari:.4f}')
        print(f'归一化互信息（NMI）: {nmi:.4f}')
        print(f"Fowlkes-Mallows Index（FMI）: {fmi:.4f}")
        if ari > max_ari:
            max_ari = ari
            best_metrics = {
                'ARI': ari,
                'NMI': nmi,
                'FMI': fmi,
                'gama': j,
            }
    # 打印ARI最大时的相关指标和参数
    print('=========================================')
    print(f'最大ARI时的相关指标和参数:')
    print(f'ARI: {best_metrics["ARI"]:.4f}')
    print(f'NMI: {best_metrics["NMI"]:.4f}')
    print(f'FMI: {best_metrics["FMI"]:.4f}')
    print(f'gama: {best_metrics["gama"]:.4f}')

