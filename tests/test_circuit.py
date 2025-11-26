import numpy as np
from src.circuit.bio_circuit import BioCircuit

def test_circuit_init():
    c = BioCircuit(num_nodes=16, topology="ring", init_weight_scale=0.01, leak=0.02, dt=0.01)
    assert c.graph.number_of_nodes() == 16

def test_step():
    c = BioCircuit(num_nodes=16, topology="ring", init_weight_scale=0.01, leak=0.02, dt=0.01)
    s0 = c.state.copy()
    c.step(external_input=np.random.randn(8, 4))
    assert np.any(c.state != s0)
