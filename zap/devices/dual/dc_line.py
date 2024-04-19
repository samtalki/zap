import numpy as np

from zap.devices.transporter import DCLine


class DualDCLine(DCLine):
    def __init__(self, line: DCLine):
        self.primal = line

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
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)

        assert data.quadratic_cost is None
        np.testing.assert_allclose(data.linear_cost, 0.0)

        # Absolutely hate 1-indexing
        z1, z2 = power[0], power[1]

        pnom = data.nominal_capacity
        pmax = data.max_power
        slack = data.slack

        u = pnom * pmax + slack

        return la.sum(la.multiply(u, la.abs(z2 - z1)))
