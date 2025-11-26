import numpy as np
from src.circuit.bio_circuit import BioCircuit
from src.models.differentiable_ic import DifferentiableIC
from src.meta_core.meta_trainer import MetaTrainer

def test_meta_loop():
    c = BioCircuit(num_nodes=9, topology="grid", init_weight_scale=0.01, leak=0.02, dt=0.01)
    m = DifferentiableIC(c)
    trainer = MetaTrainer(m, {"inner_steps": 2, "inner_lr": 0.01, "outer_lr": 0.001, "batch_tasks": 2, "episodes": 3})
    trainer.fit()
    assert m.params.shape[0] > 0
