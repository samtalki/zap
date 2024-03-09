import numpy as np
import cvxpy as cp

from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from numpy.typing import NDArray

from zap.devices.abstract import AbstractDevice, make_dynamic
from zap.util import replace_none


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

    def model_cost(self, power, angle, _, nominal_capacity=None):
        cost = cp.sum(cp.multiply(self.linear_cost, cp.abs(power[1])))
        if self.quadratic_cost is not None:
            cost += cp.sum(cp.multiply(self.quadratic_cost, cp.square(power[1])))

        return cost

    def equality_constraints(
        self, power, angle, local_variable, nominal_capacity=None, la=np
    ):
        return [power[1] + power[0]]

    def inequality_constraints(
        self, power, angle, local_variable, nominal_capacity=None, la=np
    ):
        pnom = make_dynamic(replace_none(nominal_capacity, self.nominal_capacity))
        return [
            np.multiply(self.min_power, pnom) - power[1],
            power[1] - np.multiply(self.max_power, pnom),
        ]

    def cost_grad_power(self, power, angle, local_variable, nominal_capacity=None):
        grad = np.multiply(self.linear_cost, np.sign(power))
        if self.quadratic_cost is not None:
            grad += np.multiply(2 * self.quadratic_cost, power)

        return [grad]


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

    def equality_constraints(self, power, angle, u, nominal_capacity=None, la=np):
        nominal_capacity = make_dynamic(
            replace_none(nominal_capacity, self.nominal_capacity)
        )
        susceptance = np.multiply(self.susceptance, nominal_capacity)

        eq_constraints = super().equality_constraints(power, angle, u, nominal_capacity)
        eq_constraints += [power[1] - la.multiply(susceptance, (angle[0] - angle[1]))]
        return eq_constraints
