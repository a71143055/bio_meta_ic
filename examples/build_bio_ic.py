from src.circuit.bio_circuit import BioCircuit
from src.viz.export_graph import export_graphml
import os

if __name__ == "__main__":
    c = BioCircuit(num_nodes=64, topology="hex", init_weight_scale=0.05, leak=0.01, dt=0.01)
    os.makedirs("out", exist_ok=True)
    export_graphml(c.graph, "out/circuit.graphml")
    print("Circuit built and exported to out/circuit.graphml")
