from sigil.optim.optimizer import Optimizer
import numpy as np


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, rho=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon

    def _init_variable_state(self, var):
        self.states[var] = {
            "sq_avg": np.zeros_like(var.value)  # Squared gradient average
        }

    def _update_variable(self, var):
        state = self.states[var]
        grad = var.grad

        # Update squared gradient average
        state["sq_avg"] = self.rho * state["sq_avg"] + (1 - self.rho) * (grad ** 2)

        # Update variable
        var.value -= self.learning_rate * grad / (np.sqrt(state["sq_avg"]) + self.epsilon)
        var.zero_grad()
