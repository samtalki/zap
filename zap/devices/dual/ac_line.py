import numpy as np

from zap.devices.transporter import ACLine


class DualACLine(ACLine):
    def __init__(self, line: ACLine):
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
        self.susceptance = line.susceptance

    def equality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return [angle[1] + angle[0]]

    def inequality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return []

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)

        assert data.quadratic_cost is None
        np.testing.assert_allclose(data.linear_cost, 0.0)

        # Absolutely hate 1-indexing
        z1, z2 = power[0], power[1]
        mu1, _ = angle[0], angle[1]

        b = data.susceptance
        pnom = data.nominal_capacity
        pmax = data.max_power
        slack = data.slack

        upper = la.multiply(pnom, pmax) + slack
        b_pnom = la.multiply(b, pnom)

        # For now this is
        #   = mu1 * (pnom * pmax + slack) / (b * pnom)
        #   = mu1 * pmax / b    +    mu1 * slack / (b * pnom)
        # But if pnom were a variable, we'd have to drop the second term
        # to retain convexity
        angle_term = la.multiply(mu1, upper) / b_pnom
        power_term = la.multiply(upper, z2 - z1)

        return la.sum(la.abs(power_term + angle_term))
