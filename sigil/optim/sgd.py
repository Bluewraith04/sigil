from sigil.optim.optimizer import Optimizer


class SGD(Optimizer):
    def _init_variable_state(self, var):
        # SGD has no state beyond the gradient
        pass

    def _update_variable(self, var):
        var.value -= self.learning_rate * var.grad
        var.zero_grad()
