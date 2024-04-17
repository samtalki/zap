import numpy as np

from zap.devices.abstract import make_dynamic
from zap.util import replace_none
from .transporter import Transporter


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
        nominal_capacity=None,
        capital_cost=None,
        slack=None,
        min_nominal_capacity=None,
        max_nominal_capacity=None,
    ):
        if linear_cost is None:
            linear_cost = np.zeros(capacity.shape)

        self.num_nodes = num_nodes
        self.source_terminal = source_terminal
        self.sink_terminal = sink_terminal
        self.capacity = make_dynamic(capacity)
        self.linear_cost = make_dynamic(linear_cost)
        self.quadratic_cost = make_dynamic(quadratic_cost)
        self.nominal_capacity = make_dynamic(
            replace_none(nominal_capacity, np.ones(self.num_devices))
        )
        self.capital_cost = make_dynamic(capital_cost)
        self.slack = 0.0 if slack is None else make_dynamic(slack)
        self.min_nominal_capacity = make_dynamic(min_nominal_capacity)
        self.max_nominal_capacity = make_dynamic(max_nominal_capacity)

    @property
    def min_power(self):
        return -self.capacity

    @property
    def max_power(self):
        return self.capacity


class DCLine(PowerLine):
    """A simple symmetric transporter."""

    pass
