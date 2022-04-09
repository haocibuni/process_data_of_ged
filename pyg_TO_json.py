from torch_geometric.datasets import GEDDataset
import networkx as nx
import glob
import os
import os.path as osp
import pickle
import torch
import json
dataset_name = 'LINUX'
dataset = GEDDataset("pyg_data/{}".format(dataset_name), dataset_name, train=True)
ids1 = []
gs_edge = []
gs_nodelabel = []
gs_nodenum = []
names = glob.glob('pyg_data/'+dataset_name+'/raw/'+dataset_name+'/train/*.gexf')
for i in names:
    ids1.append(int(i.split(os.sep)[-1][:-5]))
ids1.sort()
for i, idx in enumerate(ids1):

    G = nx.read_gexf(osp.join('pyg_data/'+dataset_name+'/raw/'+dataset_name+'/train/', f'{idx}.gexf'))
    mapping = {name: j for j, name in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    edge_index = list(G.edges)
    g_edge = []
    for edge in edge_index:
        g_edge.append([int(edge[0]), int(edge[1])])
    last_edge = g_edge + [[y, x] for x, y in g_edge]
    last_edge.sort()
    gs_edge.append(last_edge)

    g_label = []
    if dataset_name == 'AIDS700nef':
        for name in enumerate(G.nodes()):
            g_label.append(G.nodes().get(name)['type'])
        gs_nodelabel.append(g_label)
        gs_nodenum.append(len(g_label))
    else:
        for node_id, name in enumerate(G.nodes()):
            degree = 0
            for edge in last_edge:
                if edge[0] == name:
                    degree = degree + 1
            g_label.append(str(degree))
        gs_nodelabel.append(g_label)
        gs_nodenum.append(len(g_label))

ids2 = []
names = glob.glob('pyg_data/'+dataset_name+'/raw/'+dataset_name+'/test/*.gexf')
for i in names:
    ids2.append(int(i.split(os.sep)[-1][:-5]))
ids2.sort()

for i, idx in enumerate(ids2):
    G = nx.read_gexf(osp.join('pyg_data/'+dataset_name+'/raw/'+dataset_name+'/test/', f'{idx}.gexf'))
    mapping = {name: j for j, name in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    g_edge = []
    for edge in edge_index:
        g_edge.append([int(edge[0]), int(edge[1])])
    last_edge = g_edge + [[y, x] for x, y in g_edge]
    last_edge.sort()
    gs_edge.append(last_edge)

    g_label = []
    if dataset_name == 'AIDS700nef':
        for name in enumerate(G.nodes()):
            g_label.append(G.nodes().get(name)['type'])
        gs_nodelabel.append(g_label)
        gs_nodenum.append(len(g_label))
    else:
        for node_name in enumerate(G.nodes()):
            degree = 0
            for edge in last_edge:
                if edge[0] == name:
                    degree = degree + 1
            g_label.append(str(degree))
        gs_nodelabel.append(g_label)
        gs_nodenum.append(len(g_label))


ids = ids1 + ids2
assoc = {idx: i for i, idx in enumerate(ids)}
mat = torch.full((len(assoc), len(assoc)), float('inf'))
with open('pyg_data/'+dataset_name+'/raw/'+dataset_name+'/ged.pickle', 'rb') as f:
    obj = pickle.load(f)
    xs, ys, gs = [], [], []
    for (x, y), g in obj.items():
        xs += [assoc[x]]
        ys += [assoc[y]]
        gs += [g]
    x, y = torch.tensor(xs), torch.tensor(ys)
    g = torch.tensor(gs, dtype=torch.float)
    mat[x, y], mat[y, x] = g, g
ged = mat.tolist()
gnum = len(gs_nodenum)
train_num = int(gnum*0.8)
test_num = int(gnum - train_num)
if not os.path.exists("data/" + dataset_name):
    os.makedirs("data/" + dataset_name)

# 训练集图对
train_dataset = []
for i in range(train_num):
    for j in range(train_num):
        dic_json = {}
        dic_json["graph_1"] = gs_edge[i]
        dic_json["ged"] = int(ged[i][j])
        dic_json["graph_2"] = gs_edge[j]
        dic_json["labels_2"] = gs_nodelabel[j]
        dic_json["labels_1"] = gs_nodelabel[i]
        dic_json["id_1"] = ids[i]
        dic_json["id_2"] = ids[j]
        train_dataset.append(dic_json)
        # with open("data/MUTAG_train/" + str(i.graph["gid"]+1) + "-" + str(j.graph["gid"] + 1) + ".json", "w") as f:
        #     json.dump(dic_json, f)
        print(" " + str(i+1) + "_" + str(j+1) + "载入文件完成...")
with open("data/"+dataset_name+"/"+dataset_name+"_train.json", "w") as f:
    json.dump(train_dataset, f)
# 测试集图对
if dataset_name != 'ALKANE':
    test_dataset = []
    for i in range(train_num, gnum):
        for j in range(0, train_num):
            dic_json = {}
            dic_json["graph_1"] = gs_edge[i]
            dic_json["ged"] = int(ged[i][j])
            dic_json["graph_2"] = gs_edge[j]
            dic_json["labels_2"] = gs_nodelabel[j]
            dic_json["labels_1"] = gs_nodelabel[i]
            dic_json["id_1"] = ids[i]
            dic_json["id_2"] = ids[j]
            test_dataset.append(dic_json)
            # with open("data/MUTAG_test/" + str(i.graph["gid"] + 1) + "-" + str(j.graph["gid"] + 1) + ".json", "w") as f:
            #     json.dump(dic_json, f)
            print(" " + str(i + 1) + "_" + str(j + 1) + "载入文件完成...")
    with open("data/"+dataset_name+"/"+dataset_name+"_test.json", "w") as f:
        json.dump(test_dataset, f)