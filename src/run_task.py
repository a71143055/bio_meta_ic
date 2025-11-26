from src.meta_core.logging_setup import setup_logging
from src.cli.args import get_task_args
from src.meta_core.utils import load_yaml, set_seed
from src.meta_core.tasks import make_task
from src.circuit.bio_circuit import BioCircuit
from src.models.differentiable_ic import DifferentiableIC
from src.meta_core.meta_trainer import MetaTrainer
from src.viz.visualize import visualize_activity

def main():
    args = get_task_args()
    cfg = load_yaml(args.config)
    task_cfg = load_yaml(args.task)
    set_seed(cfg.get("seed", 42))
    logger = setup_logging()

    circuit = BioCircuit(**cfg["circuit"])
    model = DifferentiableIC(circuit)
    task = make_task(task_cfg)

    trainer = MetaTrainer(model=model, config=cfg["meta"])
    hist = trainer.fit_task(task)

    visualize_activity(hist, mode=args.mode, out_dir=cfg.get("out_dir", "out"))
    logger.info("Task run completed.")

if __name__ == "__main__":
    main()
