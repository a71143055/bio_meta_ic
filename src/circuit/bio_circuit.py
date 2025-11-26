import networkx as nx
import numpy as np
from .components import NodeParams, EdgeParams
from .dynamics import node_update, nonlinearity
from .c_backend import FastBackend

class BioCircuit:
    def __init__(self, num_nodes: int, topology: str, init_weight_scale: float, leak: float, dt: float):
        self.num_nodes = num_nodes
        self.topology = topology
        self.dt = dt
        self.graph = self._build_topology(topology)
        self.node_params = [NodeParams(leak=leak, bias=0.0) for _ in range(num_nodes)]
        self.edge_params = {}
        self.state = np.zeros((num_nodes,), dtype=np.float32)
        self.backend = FastBackend()
        self._init_weights(init_weight_scale)

    def _build_topology(self, topology):
        if topology == "grid":
            side = int(np.sqrt(self.num_nodes))
            G = nx.grid_2d_graph(side, side)
            mapping = {coord: i for i, coord in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
        elif topology == "ring":
            G = nx.cycle_graph(self.num_nodes)
        elif topology == "hex":
            side = int(np.sqrt(self.num_nodes))
            G = nx.hexagonal_lattice_graph(side, side)
            mapping = {coord: i for i, coord in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
        else:
            G = nx.gnm_random_graph(self.num_nodes, self.num_nodes * 2)
        return G

    def _init_weights(self, scale):
        for u, v in self.graph.edges():
            self.edge_params[(u, v)] = EdgeParams(weight=np.random.randn() * scale, delay=0)

    def step(self, external_input=None):
        inputs = np.zeros_like(self.state)
        for (u, v), ep in self.edge_params.items():
            inputs[v] += ep.weight * nonlinearity(self.state[u])
        if external_input is not None:
            inputs[: external_input.shape[0]] += external_input.sum(axis=1)

        self.state = self.backend.fast_update(self.state, inputs, np.array([np.float32(p.leak) for p in self.node_params]),
                                              np.array([np.float32(p.bias) for p in self.node_params]),
                                              np.float32(self.dt))

    def get_weights_vector(self):
        # Flatten edge weights in fixed order
        weights = []
        for (u, v) in self.graph.edges():
            weights.append(self.edge_params[(u, v)].weight)
        return np.array(weights, dtype=np.float32)

    def set_weights_vector(self, wvec):
        for i, (u, v) in enumerate(self.graph.edges()):
            self.edge_params[(u, v)].weight = float(wvec[i])
