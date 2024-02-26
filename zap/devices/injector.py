import cvxpy as cp
import numpy as np

from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray

from .abstract import AbstractDevice


@dataclass(kw_only=True)
class Injector(AbstractDevice):
    """A single-node device that may deposit or withdraw power from the network."""

    num_nodes: int
    terminal: NDArray
    min_power: NDArray
    max_power: NDArray
    linear_cost: NDArray
    quadratic_cost: Optional[NDArray] = None

    def __post_init__(self):
        # TODO - Add dimension checks
        pass

    @property
    def get_terminals(self):
        return self.terminal

    def model_local_constraints(self, power, angle, local_variable):
        power = power[0]

        return [
            self.min_power <= power,
            power <= self.max_power,
        ]

    def model_cost(self, power, angle, local_variable):
        power = power[0] - self.min_power

        cost = self.linear_cost.T @ power
        if self.quadratic_cost is not None:
            cost += cp.sum(cp.multiply(self.quadratic_cost, cp.square(power)))

        return cost


class Generator(Injector):
    """An Injector that can only deposit power."""

    def __init__(self, *, num_nodes, terminal, capacity, linear_cost, quadratic_cost=None):
        self.num_nodes = num_nodes
        self.terminal = terminal
        self.capacity = capacity
        self.linear_cost = linear_cost
        self.quadratic_cost = quadratic_cost

    @property
    def min_power(self):
        return np.zeros(self.capacity.shape)

    @property
    def max_power(self):
        return self.capacity


class Load(Injector):
    """An Injector that can only withdraw power."""

    def __init__(self, *, num_nodes, terminal, load, linear_cost, quadratic_cost=None):
        self.num_nodes = num_nodes
        self.terminal = terminal
        self.load = load
        self.linear_cost = linear_cost
        self.quadratic_cost = quadratic_cost

    @property
    def min_power(self):
        return -self.load

    @property
    def max_power(self):
        return np.zeros(self.load.shape)