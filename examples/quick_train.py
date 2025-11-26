import argparse
from src.meta_core.utils import load_yaml, set_seed
from src.circuit.bio_circuit import BioCircuit
from src.models.differentiable_ic import DifferentiableIC
from src.meta_core.meta_trainer import MetaTrainer
from src.meta_core.tasks import make_task
from src.viz.visualize import visualize_activity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--task", type=str, default="config/tasks/target_mimic.yaml")
    parser.add_argument("--mode", type=str, default="lines")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task_cfg = load_yaml(args.task)
    set_seed(cfg.get("seed", 42))

    circuit = BioCircuit(**cfg["circuit"])
    model = DifferentiableIC(circuit)
    task = make_task(task_cfg)

    trainer = MetaTrainer(model=model, config=cfg["meta"])
    hist = trainer.fit_task(task)
    visualize_activity(hist, mode=args.mode, out_dir=cfg.get("out_dir", "out"))
    print("Training complete. See out/ for plots.")
