from src.meta_core.logging_setup import setup_logging
from src.cli.args import get_main_args
from src.meta_core.utils import load_yaml, set_seed
from src.meta_core.meta_trainer import MetaTrainer
from src.circuit.bio_circuit import BioCircuit
from src.models.differentiable_ic import DifferentiableIC
from src.viz.export_graph import export_graphml

def main():
    args = get_main_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    logger = setup_logging()

    circuit_cfg = cfg["circuit"]
    circuit = BioCircuit(
        num_nodes=circuit_cfg["num_nodes"],
        topology=circuit_cfg["topology"],
        init_weight_scale=circuit_cfg["init_weight_scale"],
        leak=circuit_cfg["leak"],
        dt=circuit_cfg["dt"],
    )
    logger.info(f"Circuit initialized with {circuit.num_nodes} nodes, topology={circuit.topology}")

    model = DifferentiableIC(circuit)
    trainer = MetaTrainer(model=model, config=cfg["meta"])

    trainer.fit()
    out_dir = cfg.get("out_dir", "out")
    if cfg.get("visualize", {}).get("export_graphml", True):
        export_graphml(circuit.graph, f"{out_dir}/circuit.graphml")
        logger.info(f"Graph exported to {out_dir}/circuit.graphml")

if __name__ == "__main__":
    main()
