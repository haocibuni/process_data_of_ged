import networkx as nx
from ged4py.algorithm import graph_edit_dist
import re
import collections
import pickle
import glob
import tqdm
import multiprocessing.pool
from multiprocessing.pool import Pool
from torch_geometric.datasets import GEDDataset


def compute_ged(pairlist):
    g1 = pairlist[0]
    g2 = pairlist[1]
    id1 = pairlist[2]
    id2 = pairlist[3]
    ged_value = graph_edit_dist.compare(g1, g2)
    return [id1, id2, ged_value]

if __name__ == '__main__':
    multiprocessing.freeze_support()
    pool = Pool(processes=2)
    names = glob.glob('pyg_data/ALKANE/raw/ALKANE/train/*.gexf') + glob.glob('pyg_data/ALKANE/raw/ALKANE/test/*.gexf')
    pairslist = []
    for name1 in names:
        for name2 in names:
            id1 = int(re.findall("\d+", name1)[0])
            id2 = int(re.findall("\d+", name2)[0])
            g1 = nx.read_gexf(name1)
            g2 = nx.read_gexf(name2)
            pairlist = []
            pairlist.append(g1)
            pairlist.append(g2)
            pairlist.append(id1)
            pairlist.append(id2)
            pairslist.append(pairlist)
    # with tqdm.tqdm(total=len(pairslist)) as progress_bar:
    result = pool.map(compute_ged, pairslist)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()
    geds = collections.OrderedDict()
    for r in result:
        geds[(r[0], r[1])] = r[2]
    file = open("pyg_data/ALKANE/raw/ALKANE/ged.pickle", 'wb')
    pickle.dump(geds, file)
    file.close()
    dataset2 = GEDDataset("data/{}".format('ALKANE'), 'ALKANE', train=True)
    dataset1 = GEDDataset("data/{}".format('ALKANE'), 'ALKANE', train=False)
