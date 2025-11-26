import yaml
import numpy as np
import random

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def batch_tasks_sampler(n):
    # Placeholder sampler producing synthetic tasks
    tasks = []
    for _ in range(n):
        from .tasks import WaveformFollowingTask
        tasks.append(WaveformFollowingTask("sine", freq=1.0, amplitude=1.0, input_dim=2, output_dim=1, horizon=200))
    return tasks
