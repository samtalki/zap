import numpy as np
import cvxpy as cp

from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from numpy.typing import NDArray

from .abstract import AbstractDevice


@dataclass(kw_only=True)
class Transporter(AbstractDevice):
    """A two-node device that carries power between nodes.

    The net power of a transporter is always zero.
    """

    num_nodes: int
    source_terminal: NDArray
    sink_terminal: NDArray
    min_power: NDArray
    max_power: NDArray
    linear_cost: NDArray
    quadratic_cost: Optional[NDArray] = None

    def __post_init__(self):
        # TODO - Add dimension checks
        pass

    @cached_property
    def terminals(self):
        return np.column_stack((self.source_terminal, self.sink_terminal))

    def model_local_constraints(self, power, angle, local_variable):
        return [
            power[1] == -power[0],
            self.min_power <= power[1],
            power[1] <= self.max_power,
        ]

    def model_cost(self, power, angle, local_variable):
        cost = self.linear_cost.T @ power[1]
        if self.quadratic_cost is not None:
            cost += cp.sum(cp.multiply(self.quadratic_cost, cp.square(power[1])))

        return cost


class PowerLine(Transporter):
    """A simple symmetric transporter."""

    def __init__(
        self,
        *,
        num_nodes,
        source_terminal,
        sink_terminal,
        capacity,
        linear_cost=None,
        quadratic_cost=None,
    ):
        if linear_cost is None:
            linear_cost = np.zeros(capacity.shape)

        self.num_nodes = num_nodes
        self.source_terminal = source_terminal
        self.sink_terminal = sink_terminal
        self.capacity = capacity
        self.linear_cost = linear_cost
        self.quadratic_cost = quadratic_cost

    @property
    def min_power(self):
        return -self.capacity

    @property
    def max_power(self):
        return self.capacity


class DCLine(PowerLine):
    """A simple symmetric transporter."""

    pass


class ACLine(PowerLine):
    """A symmetric transporter with phase angle constraints."""

    def __init__(
        self,
        *,
        num_nodes,
        source_terminal,
        sink_terminal,
        capacity,
        susceptance,
        linear_cost=None,
        quadratic_cost=None,
    ):
        self.susceptance = susceptance

        super().__init__(
            num_nodes=num_nodes,
            source_terminal=source_terminal,
            sink_terminal=sink_terminal,
            capacity=capacity,
            linear_cost=linear_cost,
            quadratic_cost=quadratic_cost,
        )

    @property
    def is_ac(self):
        return True

    def model_local_constraints(self, power, angle, local_variable):
        constraints = [power[1] == cp.multiply(self.susceptance, (angle[0] - angle[1]))]
        constraints += super().model_local_constraints(power, angle, local_variable)
        return constraints
