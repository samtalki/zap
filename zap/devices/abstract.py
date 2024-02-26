import cvxpy as cp
import numpy as np
import scipy.sparse as sp


from abc import ABC, abstractmethod
from functools import cached_property
from numpy.typing import NDArray


class AbstractDevice(ABC):
    # Pre-defined methods

    num_nodes: int

    @property
    def num_terminals_per_device(self) -> int:
        terminals = self.get_terminals
        assert len(terminals.shape) <= 2

        if len(terminals.shape) == 1:
            return 1
        else:
            return terminals.shape[1]

    @property
    def num_devices(self) -> int:
        return self.get_terminals.shape[0]

    @cached_property
    def incidence_matrix(self):
        dimensions = (self.num_nodes, self.num_devices)

        matrices = []
        for terminal_index in range(self.num_terminals_per_device):
            vals = np.ones(self.num_devices)

            if len(self.get_terminals.shape) == 1:
                rows = self.get_terminals
            else:
                rows = self.get_terminals[:, terminal_index]

            cols = np.arange(self.num_devices)

            matrices.append(sp.csc_matrix((vals, (rows, cols)), shape=dimensions))

        return matrices

    def initialize_power(self):
        return [
            cp.Variable(self.num_devices) for _ in range(self.num_terminals_per_device)
        ]

    def initialize_angle(self):
        if self.is_ac:
            return [
                cp.Variable(self.num_devices)
                for _ in range(self.num_terminals_per_device)
            ]
        else:
            return None

    # Optionally defined methods

    @property
    def is_ac(self):
        return False

    @property
    def is_convex(self):
        return True

    def model_local_variables(self):
        return None

    @property
    def data(self):
        pass

    # Sub classes must define these methods

    @property
    @abstractmethod
    def get_terminals(self) -> NDArray:
        pass

    @abstractmethod
    def model_cost(self, power, angle, local_variable):
        pass

    @abstractmethod
    def model_local_constraints(self, power, angle, local_variable):
        pass
