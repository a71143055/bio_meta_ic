import numpy as np

def node_update(state, inputs, leak, bias, dt):
    # Simple leaky integrator activation
    return state + dt * (-leak * state + inputs + bias)

def nonlinearity(x):
    return np.tanh(x)
