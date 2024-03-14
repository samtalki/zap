import torch
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from functools import cached_property
from typing import Optional
from numpy.typing import NDArray

from zap.util import grad_or_zero


class AbstractDevice:
    # Fields

    terminals: NDArray
    num_nodes: int
    time_horizon: int

    # Overwriteable methods

    # Optional
    @property
    def is_ac(self):
        return False

    # Optional
    @property
    def is_convex(self):
        return True

    # Optional
    @property
    def data(self):
        raise NotImplementedError

    # Optional
    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return None

    def operation_cost(self, power, angle, local_variable, **kwargs):
        return NotImplementedError

    def equality_constraints(self, power, angle, local_variable, **kwargs):
        return NotImplementedError

    def inequality_constraints(self, power, angle, local_variable, **kwargs):
        return NotImplementedError

    # Pre-defined methods

    @property
    def num_terminals_per_device(self) -> int:
        terminals = self.terminals
        assert len(terminals.shape) <= 2

        if len(terminals.shape) == 1:
            return 1
        else:
            return terminals.shape[1]

    @property
    def num_devices(self) -> int:
        return self.terminals.shape[0]

    @cached_property
    def incidence_matrix(self):
        dimensions = (self.num_nodes, self.num_devices)

        matrices = []
        for terminal_index in range(self.num_terminals_per_device):
            vals = np.ones(self.num_devices)

            if len(self.terminals.shape) == 1:
                rows = self.terminals
            else:
                rows = self.terminals[:, terminal_index]

            cols = np.arange(self.num_devices)

            matrices.append(sp.csc_matrix((vals, (rows, cols)), shape=dimensions))

        return matrices

    def initialize_power(self, time_horizon: int) -> list[cp.Variable]:
        return [
            cp.Variable((self.num_devices, time_horizon))
            for _ in range(self.num_terminals_per_device)
        ]

    def initialize_angle(self, time_horizon: int) -> list[cp.Variable]:
        if self.is_ac:
            return [
                cp.Variable((self.num_devices, time_horizon))
                for _ in range(self.num_terminals_per_device)
            ]
        else:
            return None

    def operation_cost_gradients(self, power, angle, local_variable, **kwargs):
        power = [torch.tensor(p, requires_grad=True) for p in power]
        angle = (
            [torch.tensor(a, requires_grad=True) for a in angle]
            if angle is not None
            else None
        )
        local_vars = (
            [torch.tensor(lv, requires_grad=True) for lv in local_variable]
            if local_variable is not None
            else None
        )

        C = self.operation_cost(power, angle, local_vars, **kwargs, la=torch)
        if C.requires_grad:
            C.backward()

        return (
            [grad_or_zero(p).numpy() for p in power],
            [grad_or_zero(a).numpy() for a in angle] if angle is not None else None,
            [grad_or_zero(lv).numpy() for lv in local_vars]
            if local_vars is not None
            else None,
        )


def get_time_horizon(array: NDArray) -> int:
    if len(array.shape) < 2:
        return 1
    else:
        return array.shape[1]


def make_dynamic(array: Optional[NDArray]) -> NDArray:
    if (array is not None) and (len(array.shape)) == 1:
        return np.expand_dims(array, axis=1)
    else:
        return array


def _zero_like(data: Optional[list[NDArray]]) -> Optional[list[NDArray]]:
    if data is not None:
        return [np.zeros_like(d) for d in data]
    else:
        return None
