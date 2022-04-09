import networkx as nx
from ged4py.algorithm import graph_edit_dist
import json
import os
import collections
import pickle
import random
import tqdm
import multiprocessing.pool
from multiprocessing import Manager
from multiprocessing.pool import Pool
import numpy as np

def compute_ged(pairlist):
    g1 = pairlist[0]
    g2 = pairlist[1]
    ged_value = graph_edit_dist.compare(g1, g2)
    # ged_value = nx.graph_edit_distance(g1, g2)
    print(int(g1.graph["gid"] + 1), int(g2.graph["gid"] + 1), ged_value)
    return [int(g1.graph["gid"] + 1), int(g2.graph["gid"] + 1), ged_value]


if __name__ == '__main__':

    dataset_dir = "ls11_data/"
    dataset_name = "DD"
    dataset_dir = dataset_dir + dataset_name + "/" + dataset_name
    # 获取边数组，A[0]A[1]对应两个断点，int型
    A = []
    with open(dataset_dir + '_A.txt', 'r') as f:
        for line in f:
            A.append(list((line.strip('\n').split(','))))
        for i in A:
            i[1] = int(i[1].replace(" ", ""))
            i[0] = int(i[0])
    # 获取图—节点对应，i代表节点 A[i]代表i所属于的图，int型
    graph_indicator = []
    with open(dataset_dir + '_graph_indicator.txt', 'r') as f:
        for line in f:
            graph_indicator.append(int(line.strip('\n')))
    # 获取标签—节点对应，i代表节点 A[i]代表i所属于的标签，str型
    node_labels = []
    with open(dataset_dir + '_node_labels.txt', 'r') as f:
        for line in f:
            node_labels.append(str(line.strip('\n')))

    a = np.array(node_labels)
    b = np.unique(a)
    print(b)
    print(len(b))
    # 图数量
    gnum = 0
    # 图集各图节点数量
    gs_nodenum = []
    # 上一个图
    old = 1
    # 节点数量
    nodenum = 0
    for graph_indicator in graph_indicator:
        if graph_indicator == old:
            nodenum = nodenum + 1
        else:
            gs_nodenum.append(nodenum)
            nodenum = 1
            old = graph_indicator
    gs_nodenum.append(nodenum)
    gnum = len(gs_nodenum)
    # 图集的边集 三维数组
    gs_edge = []
    gold_nodenum = 0
    g_nodenum_total = 0
    for g_nodenum in gs_nodenum:
        g_edge = []
        g_min = g_nodenum_total
        g_nodenum_total = g_nodenum_total + g_nodenum
        for e in A:
            if g_min <= e[0] <= g_nodenum_total and g_min <= e[1] <= g_nodenum_total:
                g_edge.append([e[0] - gold_nodenum - 1, e[1] - gold_nodenum - 1])
        gold_nodenum = gold_nodenum + g_nodenum
        gs_edge.append(g_edge)
    # 对边集排序
    for g_edge in gs_edge:
        g_edge.sort()
    # 图集的各边集数量
    gs_edgenum = []
    for g_edge in gs_edge:
        gs_edgenum.append(len(g_edge))
    # 图集各节点的标签
    gs_nodelabel = []
    g_nodenum_total = 0
    for g_nodenum in gs_nodenum:
        g_nodelabel = node_labels[g_nodenum_total:g_nodenum + g_nodenum_total]
        gs_nodelabel.append(g_nodelabel)
        g_nodenum_total = g_nodenum_total + g_nodenum
    # 图集转华为nx
    gs = []
    for i in range(gnum):
        g = nx.Graph(gid=i)
        for j in range(gs_nodenum[i]):
            g.add_node(j, label=j, type=gs_nodelabel[i][j])
        kid = 0
        for k in gs_edge[i]:
            g.add_edge(k[0], k[1], id=kid)
            kid = kid + 1
        gs.append(g)
    for g in gs:
        if g.number_of_nodes() > 10:
            gs.remove(g)
    gs_num = len(gs)
    gs = random.sample(gs, 500)
    gs_num = len(gs)
    # 训练集和测试集数量
    train_num = int(gs_num * 0.8)
    test_num = int(gs_num - train_num)
    # 打乱gs数组顺序
    random.shuffle(gs)
    # 将gs转化为gexf
    if not os.path.exists("data/" + dataset_name + "/raw/" + dataset_name + "/train/"):
        os.makedirs("data/" + dataset_name + "/raw/" + dataset_name + "/train/")
    if not os.path.exists("data/" + dataset_name + "/raw/" + dataset_name + "/test/"):
        os.makedirs("data/" + dataset_name + "/raw/" + dataset_name + "/test/")
    for i in range(0, train_num):
        nx.write_gexf(gs[i], "data/" + dataset_name + "/raw/" + dataset_name + "/train/" + str(
            gs[i].graph["gid"] + 1) + ".gexf")
    for i in range(train_num, gs_num):
        nx.write_gexf(gs[i], "data/" + dataset_name + "/raw/" + dataset_name + "/test/" + str(
            gs[i].graph["gid"] + 1) + ".gexf")
    # 多进程计算ged矩阵并保存
    multiprocessing.freeze_support()
    pool = Pool(processes=3)
    pairslist = []
    for g1 in gs:
        for g2 in gs:
            pairlist = [g1, g2]
            pairslist.append(pairlist)
    result = pool.map(compute_ged, pairslist)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()
    geds = collections.OrderedDict()
    for r in result:
        geds[(r[0], r[1])] = r[2]
    # with tqdm.tqdm(total=len(pairslist)) as progress_bar:
    #     # 进度条函数，从中调用编码函数encode
    #     def compute_ged(pairlist):
    #         g1 = pairlist[0]
    #         g2 = pairlist[1]
    #         ged_value = graph_edit_dist.compare(g1, g2)
    #         geds[(int(g1.graph["gid"] + 1), int(g2.graph["gid"] + 1))] = ged_value
    #         progress_bar.update(1)
    #     pool.map_async(compute_ged, pairslist)
    file = open("data/" + dataset_name + "/raw/" + dataset_name + "/ged1.pickle", 'wb')
    pickle.dump(geds, file)
    file.close()
    # # 训练集图对json
    # train_dataset = []
    # for i in gs[0:train_num]:
    #     for j in gs[0:train_num]:
    #         # a = i.node()
    #         dic_json = {}
    #         dic_json["graph_1"] = gs_edge[i.graph["gid"]]
    #         dic_json["ged"] = int(graph_edit_dist.compare(i, j))
    #         dic_json["graph_2"] = gs_edge[j.graph["gid"]]
    #         dic_json["labels_2"] = gs_nodelabel[j.graph["gid"]]
    #         dic_json["labels_1"] = gs_nodelabel[i.graph["gid"]]
    #         dic_json["id_1"] = i.graph["gid"]
    #         dic_json["id_2"] = j.graph["gid"]
    #         train_dataset.append(dic_json)
    #         # with open("data/MUTAG_train/" + str(i.graph["gid"]+1) + "-" + str(j.graph["gid"] + 1) + ".json", "w") as f:
    #         #     json.dump(dic_json, f)
    #         print(" " + str(i.graph["gid"] + 1) + "_" + str(j.graph["gid"] + 1) + "载入文件完成...")
    # with open("data/"+dataset_name+"/"+dataset_name+"_train.json", "w") as f:
    #     json.dump(train_dataset, f)
    # # 测试集图对json
    # test_dataset = []
    # for i in gs[train_num:gnum]:
    #     for j in gs[0:train_num]:
    #         # a = i.node()
    #         dic_json = {}
    #         dic_json["graph_1"] = gs_edge[i.graph["gid"]]
    #         dic_json["ged"] = int(graph_edit_dist.compare(i, j))
    #         dic_json["graph_2"] = gs_edge[j.graph["gid"]]
    #         dic_json["labels_2"] = gs_nodelabel[j.graph["gid"]]
    #         dic_json["labels_1"] = gs_nodelabel[i.graph["gid"]]
    #         dic_json["id_1"] = i.graph["gid"]
    #         dic_json["id_2"] = j.graph["gid"]
    #         test_dataset.append(dic_json)
    #         # with open("data/MUTAG_test/" + str(i.graph["gid"] + 1) + "-" + str(j.graph["gid"] + 1) + ".json", "w") as f:
    #         #     json.dump(dic_json, f)
    #         print(" " + str(i.graph["gid"] + 1) + "_" + str(j.graph["gid"] + 1) + "载入文件完成...")
    # with open("data/"+dataset_name+"/"+dataset_name+"_test.json", "w") as f:
    #     json.dump(test_dataset, f)
