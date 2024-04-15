import numpy as np
import scipy.sparse as sp
from collections import namedtuple

from zap.devices.abstract import make_dynamic
from zap.util import replace_none, choose_base_modeler
from .dc_line import PowerLine


ACLineData = namedtuple(
    "ACLineData",
    [
        "min_power",
        "max_power",
        "linear_cost",
        "quadratic_cost",
        "nominal_capacity",
        "susceptance",
        "capital_cost",
    ],
)


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
        capital_cost=None,
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
            capital_cost=capital_cost,
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
            self.capital_cost,
        )

    def equality_constraints(self, power, angle, u, nominal_capacity=None, la=np):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)
        base = choose_base_modeler(la)

        susceptance = base.multiply(data.susceptance, data.nominal_capacity)

        eq_constraints = super().equality_constraints(
            power, angle, u, nominal_capacity=nominal_capacity, la=la
        )
        eq_constraints += [power[1] - la.multiply(susceptance, (angle[0] - angle[1]))]
        return eq_constraints

    def _equality_matrices(self, equalities, nominal_capacity=None, la=np):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)
        size = equalities[0].power[0].shape[1]

        time_horizon = int(size / self.num_devices)
        shaped_zeros = np.zeros((self.num_devices, time_horizon))

        susceptance = la.multiply(data.susceptance, data.nominal_capacity)
        b_mat = shaped_zeros + susceptance

        equalities[1].power[1] += sp.eye(size)
        equalities[1].angle[0] += -sp.diags(b_mat.ravel())
        equalities[1].angle[1] += sp.diags(b_mat.ravel())

        return super()._equality_matrices(equalities, nominal_capacity=nominal_capacity, la=la)

    def admm_initialize_angle_variables(self, time_horizon: int):
        return [
            np.zeros((self.num_devices, time_horizon)),
            np.zeros((self.num_devices, time_horizon)),
        ]

    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        nominal_capacity=None,
        power_weights=None,
        angle_weights=None,
        la=np,
        cvx_mode=False,
    ):
        if cvx_mode:
            return self.cvx_admm_prox_update(
                rho_power, rho_angle, power, angle, nominal_capacity=None
            )

        data = self.device_data(nominal_capacity=nominal_capacity, la=la)
        quadratic_cost = 0.0 if data.quadratic_cost is None else data.quadratic_cost
        pmax = np.multiply(data.max_power, data.nominal_capacity)
        pmin = np.multiply(data.min_power, data.nominal_capacity)
        susceptance = la.multiply(data.susceptance, data.nominal_capacity)

        assert np.sum(np.abs(data.linear_cost)) == 0.0  # TODO

        # See transporter for details on derivation
        # Here, we also have angle variables
        # However, we can write p0 and theta0 in terms of p1 and theta1
        #   p0 = -p1
        #   theta0 = theta1 + p1 / susceptance
        #
        # The solution for theta1 is:
        #   theta1 = (1/2) (angle1 + angle0) - mu * p1
        # where:
        #   mu = 1 / (2 * susceptance)

        mu = 1 / (2 * susceptance)

        # Now we can solve a minimization problem over just p1
        # with solution:
        #   (2 a + 2 rho + 2 rho mu^2) p + b sign(p)
        #   =
        #   rho (power1 - power0 + mu angle[1] - mu angle[0])
        num = rho_power * (power[1] - power[0]) + rho_angle * mu * (angle[0] - angle[1])
        denom = 2 * (quadratic_cost + rho_power + rho_angle * np.power(mu, 2))

        p1 = np.divide(num, denom)
        p1 = np.clip(p1, pmin, pmax)

        # Solve for other variables
        p0 = -p1
        theta1 = 0.5 * angle[0] + 0.5 * angle[1] - mu * p1
        theta0 = theta1 + p1 / susceptance

        return [p0, p1], [theta0, theta1]

    def cvx_admm_prox_update(self, rho_power, rho_angle, power, angle, nominal_capacity=None):
        print("Solving AC line prox update via CVX...")
        import cvxpy as cp

        data = self.device_data(nominal_capacity=nominal_capacity)
        pmax = np.multiply(data.max_power, data.nominal_capacity)
        pmin = np.multiply(data.min_power, data.nominal_capacity)
        susceptance = np.multiply(data.susceptance, data.nominal_capacity)

        p0 = cp.Variable(power[0].shape)
        p1 = cp.Variable(power[0].shape)
        theta0 = cp.Variable(angle[0].shape)
        theta1 = cp.Variable(angle[0].shape)

        objective = rho_power * (cp.sum_squares(p0 - power[0]) + cp.sum_squares(p1 - power[1]))
        objective += rho_angle * (
            cp.sum_squares(theta0 - angle[0]) + cp.sum_squares(theta1 - angle[1])
        )

        constraints = [
            p1 + p0 == 0,
            p1 == cp.multiply(susceptance, (theta0 - theta1)),
            p1 <= pmax,
            p1 >= pmin,
        ]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.MOSEK)

        return [p0.value, p1.value], [theta0.value, theta1.value]
