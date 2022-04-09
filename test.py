import pickle
from multiprocessing import Pool, Manager
import os
import time
import pandas as pd
from tqdm import tqdm
import collections
import glob
import networkx as nx
from ged4py.algorithm import graph_edit_dist
import re
from torch_geometric.datasets import GEDDataset, QM9
from multiprocessing import Process

with open('data/AIDS700/raw/AIDS700/ged.pickle', 'rb') as f:
    obj = pickle.load(f)
# with open('data/NCI700/raw/NCI700/ged.pickle', 'rb') as f:
#     obj = pickle.load(f)
# print('test')
# dataset_name = 'NCI700'
# dataset = GEDDataset("data/{}".format(dataset_name), dataset_name, train=True)
# print('test')
geds = collections.OrderedDict()
names = glob.glob('data/NCI700/raw/NCI700/test/*.gexf')+glob.glob('data/NCI700/raw/NCI700/train/*.gexf')
gs = []
types = []
maxn = 0
for name1 in names:
    g = nx.read_gexf(name1)
    if len(g.nodes) > maxn:
        maxn = len(g.nodes)
#     for node in g.nodes:
#         type = g.nodes().get(node)['type']
#         types.append(type)
# typeset = list(set(types))
print(maxn)
# for name1 in names:
#     for name2 in names:
#         id1 = int(re.findall("\d+", name1)[2])
#         id2 = int(re.findall("\d+", name2)[2])
#         g1 = nx.read_gexf(name1)
#         g1.name = str(id1)
#         g2 = nx.read_gexf(name2)
#         g2.name = str(id2)
#         ged_value = ged(g1, g2, 'vj')
#         ged_value = nx.graph_edit_distance(g1, g2)
#         geds[(id1, id2)] = ged_value
#         print(id1, id2, ged_value)
# file = open('ged1.pickle', 'wb')
# pickle.dump(geds, file)
# file.close()
print('test')
