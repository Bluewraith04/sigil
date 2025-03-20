from abc import ABC, abstractmethod


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.states = {}
        self.variables = []

    def register_vars(self, *variables):
        """Initialize state for new variables."""
        for var in variables:
            if var not in self.states:
                self.variables.append(var)
                self._init_variable_state(var)

    @abstractmethod
    def _init_variable_state(self, var):
        """Initialize optimizer state for a variable (e.g. momentum buffers)"""
        pass

    def update(self):
        """Update all registered variables using their gradients and internal state."""
        for var in self.variables:
            self._update_variable(var)

    @abstractmethod
    def _update_variable(self, var):
        """Update a variable using its gradient and internal state"""
        pass
