# src/viz/export_for_blender.py
import json
import numpy as np
import networkx as nx
import os

def export_graph_for_blender(G: nx.Graph, node_positions: dict = None, out_dir: str = "out"):
    os.makedirs(out_dir, exist_ok=True)
    # node_positions: {node: (x,y,z)}. If None, use spring layout 2D and z=0
    if node_positions is None:
        pos2d = nx.spring_layout(G, seed=42)
        node_positions = {n: (float(x), float(y), 0.0) for n, (x, y) in pos2d.items()}

    nodes = [{"id": int(n), "pos": node_positions[n]} for n in G.nodes()]
    edges = [{"u": int(u), "v": int(v)} for u, v in G.edges()]

    with open(f"{out_dir}/circuit_graph.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
    print("Exported:", f"{out_dir}/circuit_graph.json")

def export_node_states(states: np.ndarray, out_dir: str = "out"):
    # states: shape (T, N) where N = number of nodes
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(f"{out_dir}/node_states.csv", states, delimiter=",")
    print("Exported:", f"{out_dir}/node_states.csv")

if __name__ == "__main__":
    # 예시 사용법
    G = nx.cycle_graph(16)
    export_graph_for_blender(G)
    T = 200
    N = G.number_of_nodes()
    t = np.linspace(0, 2*np.pi, T)
    states = np.sin(np.outer(t, np.linspace(1, 3, N)))
    export_node_states(states)
