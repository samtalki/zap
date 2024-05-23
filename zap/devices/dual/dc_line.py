import numpy as np

from zap.devices.transporter import DCLine
from zap.util import envelope_variable, use_envelope


class DualDCLine(DCLine):
    def __init__(self, line: DCLine, max_price=None, **kwargs):
        self.primal = line
        self.max_price = max_price

        self.num_nodes = line.num_nodes
        self.source_terminal = line.source_terminal
        self.sink_terminal = line.sink_terminal
        self.capacity = line.capacity
        self.linear_cost = line.linear_cost
        self.quadratic_cost = line.quadratic_cost
        self.nominal_capacity = line.nominal_capacity
        self.capital_cost = line.capital_cost
        self.slack = line.slack
        self.min_nominal_capacity = line.min_nominal_capacity
        self.max_nominal_capacity = line.max_nominal_capacity

    def equality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return []

    def inequality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return []

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)

        assert self.quadratic_cost is None
        np.testing.assert_allclose(self.linear_cost, 0.0)

        # Absolutely hate 1-indexing
        z1, z2 = power[0], power[1]

        pnom = nominal_capacity
        pmax = self.max_power
        slack = self.slack

        # Cost function is
        # = (pnom*pmax + slack) * |z2 - z1|
        # = slack * |z_diff| + pmax * |pnom * z_diff|

        z_diff = z2 - z1

        if use_envelope(envelope):
            print("Envelope relaxation applied to dual DC line.")
            env, lower, upper = envelope
            lb, ub = lower["nominal_capacity"], upper["nominal_capacity"]
            max_z_diff = 2 * self.max_price
            z_diff_pnom = envelope_variable(pnom, z_diff, lb, ub, -max_z_diff, max_z_diff, *env)

        else:
            z_diff_pnom = la.multiply(z_diff, pnom)

        return la.sum(la.multiply(slack, la.abs(z_diff)) + la.multiply(pmax, la.abs(z_diff_pnom)))
