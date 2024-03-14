import torch
import numpy as np

from collections import namedtuple
from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from numpy.typing import NDArray

from zap.devices.abstract import AbstractDevice, make_dynamic
from zap.util import replace_none


TransporterData = namedtuple(
    "TransporterData",
    [
        "min_power",
        "max_power",
        "linear_cost",
        "quadratic_cost",
        "nominal_capacity",
    ],
)

ACLineData = namedtuple(
    "ACLineData",
    [
        "min_power",
        "max_power",
        "linear_cost",
        "quadratic_cost",
        "nominal_capacity",
        "susceptance",
    ],
)


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
    nominal_capacity: Optional[NDArray] = None

    def __post_init__(self):
        # Reshape arrays
        self.min_power = make_dynamic(self.min_power)
        self.max_power = make_dynamic(self.max_power)
        self.linear_cost = make_dynamic(self.linear_cost)
        self.quadratic_cost = make_dynamic(self.quadratic_cost)
        self.nominal_capacity = make_dynamic(
            replace_none(self.nominal_capacity, np.ones(self.num_devices))
        )

        # TODO - Add dimension checks
        pass

    @cached_property
    def terminals(self):
        return np.column_stack((self.source_terminal, self.sink_terminal))

    @property
    def time_horizon(self):
        return 0  # Static device

    def _device_data(self, nominal_capacity=None):
        return TransporterData(
            self.min_power,
            self.max_power,
            self.linear_cost,
            self.quadratic_cost,
            make_dynamic(replace_none(nominal_capacity, self.nominal_capacity)),
        )

    def equality_constraints(self, power, angle, _, nominal_capacity=None, la=np):
        return [power[1] + power[0]]

    def inequality_constraints(self, power, angle, _, nominal_capacity=None, la=np):
        pnom = make_dynamic(replace_none(nominal_capacity, self.nominal_capacity))
        return [
            np.multiply(self.min_power, pnom) - power[1],
            power[1] - np.multiply(self.max_power, pnom),
        ]

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np):
        linear_cost = self.linear_cost
        quadratic_cost = self.quadratic_cost

        if la == torch:
            linear_cost = torch.tensor(linear_cost)
            quadratic_cost = (
                torch.tensor(quadratic_cost) if quadratic_cost is not None else None
            )

        cost = la.sum(la.multiply(linear_cost, la.abs(power[1])))
        if quadratic_cost is not None:
            cost += la.sum(la.multiply(quadratic_cost, la.square(power[1])))

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
        nominal_capacity=None,
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
        nominal_capacity=None,
    ):
        self.susceptance = make_dynamic(susceptance)

        super().__init__(
            num_nodes=num_nodes,
            source_terminal=source_terminal,
            sink_terminal=sink_terminal,
            capacity=capacity,
            linear_cost=linear_cost,
            quadratic_cost=quadratic_cost,
            nominal_capacity=nominal_capacity,
        )

    @property
    def is_ac(self):
        return True

    def _device_data(self, nominal_capacity=None):
        return ACLineData(
            self.min_power,
            self.max_power,
            self.linear_cost,
            self.quadratic_cost,
            make_dynamic(replace_none(nominal_capacity, self.nominal_capacity)),
            self.susceptance,
        )

    def equality_constraints(self, power, angle, u, nominal_capacity=None, la=np):
        nominal_capacity = make_dynamic(
            replace_none(nominal_capacity, self.nominal_capacity)
        )
        susceptance = np.multiply(self.susceptance, nominal_capacity)

        eq_constraints = super().equality_constraints(power, angle, u, nominal_capacity)
        eq_constraints += [power[1] - la.multiply(susceptance, (angle[0] - angle[1]))]
        return eq_constraints
