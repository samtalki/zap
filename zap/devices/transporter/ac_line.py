import numpy as np
import scipy.sparse as sp
import torch
from collections import namedtuple

from zap.devices.abstract import make_dynamic
from zap.util import replace_none, envelope_variable, use_envelope
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
        "reconductoring_cost",
        "reconductoring_threshold",
        "slack",
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
        slack=None,
        min_nominal_capacity=None,
        max_nominal_capacity=None,
        reconductoring_cost=None,
        reconductoring_threshold=None,
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
            slack=slack,
            min_nominal_capacity=min_nominal_capacity,
            max_nominal_capacity=max_nominal_capacity,
            reconductoring_cost=reconductoring_cost,
            reconductoring_threshold=reconductoring_threshold,
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
            self.reconductoring_cost,
            self.reconductoring_threshold,
            self.slack,
        )

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(self, power, angle, u, nominal_capacity=None, la=np, envelope=None):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)

        # Regular transporter constraints
        eq_constraints = super().equality_constraints(
            power, angle, u, nominal_capacity=nominal_capacity, la=la
        )

        # Linearized power flow constraints
        angle_diff = angle[0] - angle[1]

        if use_envelope(envelope):  # When line is plannable
            print("Envelope relaxation applied to AC line.")
            pnom_dtheta = self.get_envelope_variable(*envelope, data, angle_diff)
        else:
            pnom_dtheta = la.multiply(angle_diff, data.nominal_capacity)

        eq_constraints += [power[1] - la.multiply(data.susceptance, pnom_dtheta)]

        return eq_constraints

    def get_envelope_variable(self, env, lb, ub, data, angle_diff):
        # Get lower and upper bounds for the angle difference
        # dtheta_max = fmax / b = pnom * pmax / b
        ub_angle_diff = ub["nominal_capacity"] * data.max_power / data.susceptance
        lb_angle_diff = -ub_angle_diff

        return envelope_variable(
            data.nominal_capacity,
            angle_diff,
            lb["nominal_capacity"],
            ub["nominal_capacity"],
            lb_angle_diff,
            ub_angle_diff,
            *env,
        )

    # ====
    # DIFFERENTIATION
    # ====

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

    # ====
    # ADMM FUNCTIONS
    # ====

    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        nominal_capacity=None,
        power_weights=None,
        angle_weights=None,
        cvx_mode=False,
        data=None,
    ):
        assert data is not None
        # machine = power[0].device
        # data = self.device_data(nominal_capacity=nominal_capacity, la=la)

        if cvx_mode:
            return self.cvx_admm_prox_update(
                rho_power, rho_angle, power, angle, nominal_capacity=None
            )

        quadratic_cost = 0.0 if data.quadratic_cost is None else data.quadratic_cost
        pmax = torch.multiply(data.max_power, data.nominal_capacity) + data.slack
        pmin = torch.multiply(data.min_power, data.nominal_capacity) - data.slack
        susceptance = torch.multiply(data.susceptance, data.nominal_capacity)

        # assert torch.sum(torch.abs(data.linear_cost)) == 0.0  # TODO

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

        mu = torch.divide(1, 2 * susceptance)

        # Now we can solve a minimization problem over just p1
        # with solution:
        #   (2 a + 2 rho + 2 rho mu^2) p + b sign(p)
        #   =
        #   rho (power1 - power0 + mu angle[1] - mu angle[0])
        num = rho_power * (power[1] - power[0]) + rho_angle * mu * (angle[0] - angle[1])
        denom = 2 * (quadratic_cost + rho_power + rho_angle * torch.pow(mu, 2))

        p1 = torch.divide(num, denom)
        p1 = torch.clip(p1, pmin, pmax)

        # Solve for other variables
        p0 = -p1
        theta1 = 0.5 * angle[0] + 0.5 * angle[1] - mu * p1
        theta0 = theta1 + p1 / susceptance

        return [p0, p1], [theta0, theta1]

    def cvx_admm_prox_update(self, rho_power, rho_angle, power, angle, nominal_capacity=None):
        print("Solving AC line prox update via CVX...")
        import cvxpy as cp

        data = self.device_data(nominal_capacity=nominal_capacity)
        pmax = np.multiply(data.max_power, data.nominal_capacity) + data.slack
        pmin = np.multiply(data.min_power, data.nominal_capacity) - data.slack
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
