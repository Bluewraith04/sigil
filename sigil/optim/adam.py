from sigil.optim.optimizer import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step counter

    def _init_variable_state(self, var):
        # Initialize first and second moment estimates
        self.states[var] = {
            "m": np.zeros_like(var.value),  # First moment (mean)
            "v": np.zeros_like(var.value)  # Second moment (variance)
        }

    def update(self):
        self.t += 1
        super().update()

    def _update_variable(self, var):
        state = self.states[var]
        grad = var.grad

        # Update moments
        state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
        state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad ** 2)

        # Bias-corrected moments
        m_hat = state["m"] / (1 - self.beta1 ** self.t)
        v_hat = state["v"] / (1 - self.beta2 ** self.t)

        # Update variable
        var.value -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        var.zero_grad()
