import numpy as np

def mse_loss(outputs, targets):
    return float(np.mean((outputs - targets) ** 2))
