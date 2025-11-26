import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
from tqdm import trange
from .utils import batch_tasks_sampler
from ..models.losses import mse_loss

@dataclass
class MetaConfig:
    inner_steps: int
    inner_lr: float
    outer_lr: float
    batch_tasks: int
    episodes: int

class MetaTrainer:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.cfg = MetaConfig(**config)

    def fit(self):
        # Generic meta loop without specific tasks (sanity run)
        for _ in trange(self.cfg.episodes, desc="Meta episodes"):
            tasks = batch_tasks_sampler(self.cfg.batch_tasks)
            grads_accum = None
            for task in tasks:
                grads = self._inner_loop(task)
                grads_accum = grads if grads_accum is None else [g1 + g2 for g1, g2 in zip(grads_accum, grads)]
            self.model.outer_update(grads_accum, self.cfg.outer_lr)

    def fit_task(self, task):
        history = {"loss": [], "outputs": [], "targets": []}
        for _ in trange(self.cfg.episodes, desc="Meta episodes"):
            grads_accum = None
            for _ in range(self.cfg.batch_tasks):
                grads, outputs, targets = self._inner_loop(task, return_outputs=True)
                loss = mse_loss(outputs, targets)
                history["loss"].append(loss)
                history["outputs"].append(outputs)
                history["targets"].append(targets)
                grads_accum = grads if grads_accum is None else [g1 + g2 for g1, g2 in zip(grads_accum, grads)]
            self.model.outer_update(grads_accum, self.cfg.outer_lr)
        return history

    def _inner_loop(self, task, return_outputs=False):
        # Clone fast weights
        fast_params = self.model.get_params_copy()
        outputs, targets = None, None
        for _ in range(self.cfg.inner_steps):
            inputs, targets = task.sample_batch()
            outputs = self.model.forward(inputs, fast_params)
            loss = mse_loss(outputs, targets)
            grads = self.model.backward(inputs, targets, outputs, fast_params)
            # SGD step on fast params
            for p, g in zip(fast_params, grads):
                p -= self.cfg.inner_lr * g
        # Compute meta-gradient w.r.t. base params by difference
        meta_grads = [p_fast - p_base for p_fast, p_base in zip(fast_params, self.model.params)]
        if return_outputs:
            return meta_grads, outputs, targets
        return meta_grads
