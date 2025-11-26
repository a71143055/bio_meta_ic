import numpy as np

class TargetMimicTask:
    def __init__(self, target_array: np.ndarray, input_dim: int, output_dim: int, horizon: int, noise_std: float):
        self.target = target_array
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.noise_std = noise_std
        self.ptr = 0

    def sample_batch(self, batch_size: int = 32):
        idx = (np.arange(batch_size) + self.ptr) % self.horizon
        self.ptr = (self.ptr + batch_size) % self.horizon
        inputs = np.random.randn(batch_size, self.input_dim) * self.noise_std
        targets = self.target[idx, :self.output_dim]
        return inputs, targets

class WaveformFollowingTask:
    def __init__(self, waveform: str, freq: float, amplitude: float, input_dim: int, output_dim: int, horizon: int):
        self.waveform = waveform
        self.freq = freq
        self.amplitude = amplitude
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.t = 0

    def sample_batch(self, batch_size: int = 32):
        t = (np.arange(batch_size) + self.t) / self.horizon
        self.t = (self.t + batch_size) % self.horizon
        if self.waveform == "sine":
            y = self.amplitude * np.sin(2 * np.pi * self.freq * t)
        elif self.waveform == "square":
            y = self.amplitude * np.sign(np.sin(2 * np.pi * self.freq * t))
        else:
            y = self.amplitude * (2 * (t * self.freq - np.floor(0.5 + t * self.freq)))
        targets = np.stack([y] * self.output_dim, axis=1)
        inputs = np.random.randn(batch_size, self.input_dim) * 0.01
        return inputs, targets

def make_task(cfg):
    import numpy as np
    name = cfg["task_name"]
    if name == "target_mimic":
        target = np.load(cfg["target_path"])
        return TargetMimicTask(
            target_array=target,
            input_dim=cfg["input_dim"],
            output_dim=cfg["output_dim"],
            horizon=cfg["horizon"],
            noise_std=cfg.get("noise_std", 0.01),
        )
    elif name == "waveform_following":
        return WaveformFollowingTask(
            waveform=cfg["waveform"],
            freq=cfg["freq"],
            amplitude=cfg["amplitude"],
            input_dim=cfg["input_dim"],
            output_dim=cfg["output_dim"],
            horizon=cfg["horizon"],
        )
    else:
        raise ValueError(f"Unknown task {name}")
