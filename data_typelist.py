import glob
import networkx as nx
names = glob.glob('data/NCI700/raw/NCI700/test/*.gexf')+glob.glob('data/NCI700/raw/NCI700/train/*.gexf')
gs = []
types = []
for name in names:
    g = nx.read_gexf(name)
    for node in g.nodes:
        type = g.nodes().get(node)['type']
        types.append(type)
typeset = list(set(types))
print(typeset)