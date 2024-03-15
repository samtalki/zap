import torch
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from functools import cached_property
from typing import Optional
from numpy.typing import NDArray

from zap.util import grad_or_zero, torchify


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
    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return None

    def operation_cost(self, power, angle, local_variable, **kwargs):
        return NotImplementedError

    def equality_constraints(self, power, angle, local_variable, **kwargs):
        return NotImplementedError

    def inequality_constraints(self, power, angle, local_variable, **kwargs):
        return NotImplementedError

    def _device_data(self, **kwargs):
        return NotImplementedError

    # Properties

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

    # Modeling Tools

    def device_data(self, la=np, **kwargs):
        data = self._device_data(**kwargs)
        if la == torch:
            data = type(data)(*[torchify(x) for x in data])

        return data

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

    # Differentiation Tools

    def operation_cost_gradients(self, power, angle, local_variables, **kwargs):
        power = torchify(power, requires_grad=True)
        angle = torchify(angle, requires_grad=True)
        local_variables = torchify(local_variables, requires_grad=True)

        C = self.operation_cost(power, angle, local_variables, **kwargs, la=torch)
        if C.requires_grad:
            C.backward()

        return (
            grad_or_zero(power, to_numpy=True),
            grad_or_zero(angle, to_numpy=True),
            grad_or_zero(local_variables, to_numpy=True),
        )

    def lagrangian(
        self, power, angle, local_vars, equality_duals, inequality_duals, la=np, **kwargs
    ):
        # Cost term
        L = self.operation_cost(power, angle, local_vars, **kwargs, la=la)

        # Constraint terms
        eqs = self.equality_constraints(power, angle, local_vars, **kwargs, la=la)
        eq_terms = [
            la.sum(la.multiply(constraint, dual)) for constraint, dual in zip(eqs, equality_duals)
        ]

        ineqs = self.inequality_constraints(power, angle, local_vars, **kwargs, la=la)
        ineq_terms = [
            la.sum(la.multiply(constraint, dual))
            for constraint, dual in zip(ineqs, inequality_duals)
        ]

        for term in eq_terms + ineq_terms:
            L += term

        return L

    def lagrangian_gradients(
        self, power, angle, local_vars, equality_duals, inequality_duals, **kwargs
    ):
        power = torchify(power, requires_grad=True)
        angle = torchify(angle, requires_grad=True)
        local_vars = torchify(local_vars, requires_grad=True)

        equality_duals = torchify(equality_duals)
        inequality_duals = torchify(inequality_duals)

        L = self.lagrangian(
            power, angle, local_vars, equality_duals, inequality_duals, la=torch, **kwargs
        )

        if L.requires_grad:
            L.backward()

        return (
            grad_or_zero(power, to_numpy=True),
            grad_or_zero(angle, to_numpy=True),
            grad_or_zero(local_vars, to_numpy=True),
        )
