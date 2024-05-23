import numpy as np

from zap.devices.transporter import ACLine
from zap.util import envelope_variable, use_envelope


class DualACLine(ACLine):
    def __init__(self, line: ACLine, max_price=None, **kwargs):
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
        self.susceptance = line.susceptance

    def equality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return [angle[1] + angle[0]]

    def inequality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return []

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)

        assert self.quadratic_cost is None
        np.testing.assert_allclose(self.linear_cost, 0.0)

        # Absolutely hate 1-indexing
        z1, z2 = power[0], power[1]
        mu1, _ = angle[0], angle[1]

        b = self.susceptance
        pnom = nominal_capacity
        pmax = self.max_power
        slack = self.slack

        # The cost function is
        #   = | (z2 - z1) * upper + mu1 * upper / (b * pnom) |
        # where
        #   upper = pnom * pmax + slack
        if not use_envelope(envelope):
            upper = la.multiply(pnom, pmax) + slack
            power_term = la.multiply(upper, z2 - z1)

            b_pnom = la.multiply(b, pnom)
            angle_term = la.multiply(mu1, upper) / b_pnom

        else:
            print("Envelope relaxation applied to dual AC line.")
            env, lower, upper = envelope

            lb, ub = lower["nominal_capacity"], upper["nominal_capacity"]
            max_z_diff = 2 * self.max_price  # 2x because it's a difference between prices

            # Power term
            z_diff = z2 - z1
            z_diff_pnom = envelope_variable(pnom, z_diff, lb, ub, -max_z_diff, max_z_diff, *env)

            # upper * |z_diff| = |slack * z_diff + pmax * pnom_z_diff|
            power_term = la.multiply(slack, z_diff) + la.multiply(pmax, z_diff_pnom)

            # Angle term
            # The second term expands to
            #   = mu1 * (pnom * pmax + slack) / (b * pnom)
            #   = mu1 * pmax / b    +    mu1 * slack / (b * pnom)
            # Since pnom is a variable, we need to drop the non-convex fractional term:
            #   - mu1 * slack / (b * pnom) term
            # This is, of course, incorrect, but it's a sufficient approximation.
            angle_term = la.multiply(mu1, pmax) / b

        return la.sum(la.abs(power_term + angle_term))
