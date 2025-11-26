import os
import networkx as nx

def export_graphml(G, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_graphml(G, path)
