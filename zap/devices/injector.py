import torch
import numpy as np

from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray

from zap.util import replace_none
from .abstract import AbstractDevice, get_time_horizon, make_dynamic


@dataclass(kw_only=True)
class Injector(AbstractDevice):
    """A single-node device that may deposit or withdraw power from the network."""

    num_nodes: int
    terminal: NDArray
    min_power: NDArray
    max_power: NDArray
    linear_cost: NDArray
    quadratic_cost: Optional[NDArray] = None
    nominal_capacity: Optional[NDArray] = None

    def __post_init__(self):
        if self.nominal_capacity is None:
            self.nominal_capacity = np.ones(self.num_devices)

        # Reshape arrays
        self.min_power = make_dynamic(self.min_power)
        self.max_power = make_dynamic(self.max_power)
        self.linear_cost = make_dynamic(self.linear_cost)
        self.quadratic_cost = make_dynamic(self.quadratic_cost)
        self.nominal_capacity = make_dynamic(self.nominal_capacity)

        # TODO - Add dimension checks
        pass

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        return get_time_horizon(self.min_power)

    def equality_constraints(self, power, angle, _, nominal_capacity=None, la=np):
        return []

    def inequality_constraints(self, power, angle, _, nominal_capacity=None, la=np):
        pnom = make_dynamic(replace_none(nominal_capacity, self.nominal_capacity))
        power = power[0]

        return [
            np.multiply(self.min_power, pnom) - power,
            power - np.multiply(self.max_power, pnom),
        ]

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np):
        pnom = make_dynamic(replace_none(nominal_capacity, self.nominal_capacity))
        min_power = self.min_power
        linear_cost = self.linear_cost
        quadratic_cost = self.quadratic_cost

        if la == torch:
            pnom = torch.tensor(pnom)
            min_power = torch.tensor(min_power)
            linear_cost = torch.tensor(linear_cost)
            quadratic_cost = (
                torch.tensor(quadratic_cost) if quadratic_cost is not None else None
            )
            la_base = torch
        else:
            la_base = np

        power = power[0] - la_base.multiply(min_power, pnom)

        cost = la.sum(la.multiply(linear_cost, power))
        if quadratic_cost is not None:
            cost += la.sum(la.multiply(quadratic_cost, la.square(power)))

        return cost


class Generator(Injector):
    """An Injector that can only deposit power."""

    def __init__(
        self,
        *,
        num_nodes,
        terminal,
        dynamic_capacity,
        linear_cost,
        quadratic_cost=None,
        nominal_capacity=None,
    ):
        self.num_nodes = num_nodes
        self.terminal = terminal

        if nominal_capacity is None:
            nominal_capacity = np.ones(self.num_devices)

        self.dynamic_capacity = make_dynamic(dynamic_capacity)
        self.nominal_capacity = make_dynamic(nominal_capacity)
        self.linear_cost = make_dynamic(linear_cost)
        self.quadratic_cost = make_dynamic(quadratic_cost)

        # TODO - Add dimension checks
        pass

    @property
    def min_power(self):
        return np.zeros(self.dynamic_capacity.shape)

    @property
    def max_power(self):
        return self.dynamic_capacity


class Load(Injector):
    """An Injector that can only withdraw power."""

    def __init__(self, *, num_nodes, terminal, load, linear_cost, quadratic_cost=None):
        self.num_nodes = num_nodes
        self.terminal = terminal

        self.nominal_capacity = make_dynamic(np.ones(self.num_devices))
        self.load = make_dynamic(load)
        self.linear_cost = make_dynamic(linear_cost)
        self.quadratic_cost = make_dynamic(quadratic_cost)

    @property
    def min_power(self):
        return -self.load

    @property
    def max_power(self):
        return np.zeros(self.load.shape)
