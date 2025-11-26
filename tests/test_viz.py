from src.viz.export_graph import export_graphml
from src.circuit.bio_circuit import BioCircuit
import os

def test_export_graphml(tmp_path):
    c = BioCircuit(num_nodes=9, topology="grid", init_weight_scale=0.01, leak=0.02, dt=0.01)
    out = tmp_path / "c.graphml"
    export_graphml(c.graph, str(out))
    assert os.path.exists(out)
