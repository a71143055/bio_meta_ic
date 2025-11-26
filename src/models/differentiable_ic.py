import numpy as np

class DifferentiableIC:
    def __init__(self, circuit):
        self.circuit = circuit
        # Base parameters are the edge weights (vectorized)
        self.params = circuit.get_weights_vector()

    def forward(self, inputs, params=None):
        if params is not None:
            self.circuit.set_weights_vector(params)
        # Roll the circuit over a single step with external inputs
        self.circuit.step(external_input=inputs)
        # Readout: take top-k nodes as outputs aggregated
        k = 2
        out = self.circuit.state[:k].reshape(1, -1)
        # Broadcast to batch size
        return np.repeat(out, inputs.shape[0], axis=0)

    def backward(self, inputs, targets, outputs, params=None):
        # Simple gradient: dLoss/dw approximated via finite differences
        eps = 1e-3
        base = params if params is not None else self.params
        grads = np.zeros_like(base)
        for i in range(base.shape[0]):
            w_plus = base.copy()
            w_minus = base.copy()
            w_plus[i] += eps
            w_minus[i] -= eps
            y_plus = self.forward(inputs, w_plus)
            y_minus = self.forward(inputs, w_minus)
            loss_plus = ((y_plus - targets) ** 2).mean()
            loss_minus = ((y_minus - targets) ** 2).mean()
            grads[i] = (loss_plus - loss_minus) / (2 * eps)
        return [grads]

    def get_params_copy(self):
        return [self.params.copy()]

    def outer_update(self, grads_list, lr):
        # Single vector param
        grad = grads_list[0]
        self.params -= lr * grad
        self.circuit.set_weights_vector(self.params)
