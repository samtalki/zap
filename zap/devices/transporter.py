import numpy as np
import scipy.sparse as sp

from collections import namedtuple
from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from numpy.typing import NDArray

from zap.devices.abstract import AbstractDevice, make_dynamic
from zap.util import replace_none, choose_base_modeler


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
    capital_cost: Optional[NDArray] = None

    def __post_init__(self):
        # Reshape arrays
        self.min_power = make_dynamic(self.min_power)
        self.max_power = make_dynamic(self.max_power)
        self.linear_cost = make_dynamic(self.linear_cost)
        self.quadratic_cost = make_dynamic(self.quadratic_cost)
        self.nominal_capacity = make_dynamic(
            replace_none(self.nominal_capacity, np.ones(self.num_devices))
        )
        self.capital_cost = make_dynamic(self.capital_cost)

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
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)
        base = choose_base_modeler(la)

        return [
            base.multiply(data.min_power, data.nominal_capacity) - power[1],
            power[1] - base.multiply(data.max_power, data.nominal_capacity),
        ]

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)

        cost = la.sum(la.multiply(data.linear_cost, la.abs(power[1])))
        if data.quadratic_cost is not None:
            cost += la.sum(la.multiply(data.quadratic_cost, la.square(power[1])))

        return cost

    def _equality_matrices(self, equalities, nominal_capacity=None, la=np):
        size = equalities[0].power[0].shape[1]

        equalities[0].power[0] += sp.eye(size)
        equalities[0].power[1] += sp.eye(size)

        return equalities

    def _inequality_matrices(self, inequalities, nominal_capacity=None, la=np):
        size = inequalities[0].power[0].shape[1]

        inequalities[0].power[1] += -sp.eye(size)
        inequalities[1].power[1] += sp.eye(size)

        return inequalities

    def scale_costs(self, scale):
        self.linear_cost /= scale
        if self.quadratic_cost is not None:
            self.quadratic_cost /= scale
        if self.capital_cost is not None:
            self.capital_cost /= scale

    def scale_power(self, scale):
        self.nominal_capacity /= scale

    def admm_initialize_power_variables(self, time_horizon: int):
        return [
            np.zeros((self.num_devices, time_horizon)),
            np.zeros((self.num_devices, time_horizon)),
        ]

    def admm_initialize_angle_variables(self, time_horizon: int):
        return None

    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        nominal_capacity=None,
        la=np,
        power_weights=None,
        angle_weights=None,
    ):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)
        quadratic_cost = 0.0 if data.quadratic_cost is None else data.quadratic_cost
        pmax = np.multiply(data.max_power, data.nominal_capacity)
        pmin = np.multiply(data.min_power, data.nominal_capacity)

        Dp2 = [np.power(p, 2) for p in power_weights]

        assert angle is None

        # TODO
        # This shouldn't be too hard, we just solve the two cases (p1 < 0) and (p1 > 0)
        # Then project onto [pmin, 0] and [0, pmax]
        # Then pick the better of the two
        assert np.sum(np.abs(data.linear_cost)) == 0.0

        # Problem is
        #     min_p    a p1^2 + b |p1|
        #              + (rho/2) ||D0 (p0 - power0)||_2^2 + (rho/2) ||D1 (p1 - power1)||_2^2
        #              + {p1 box constraints} + {p0 + p1 = 0}
        #
        # Setting p0 = -p1, we remove the equality constraint and reformulate as
        #     min_p1   a p1^2 + b |p1|
        #              + (rho/2) ||D0 (-p1 - power0) ||_2^2 + (rho/2) ||D1 (p1 - power1) ||_2^2
        #              + {p1 box constraints}
        #
        # The objective derivative is then
        #     2 a p1 + b sign(p1) - rho D0^2 (-p1 - power0) + rho D1^2 (p1 - power1),
        #
        # which is solved by
        #     2 a p1 + b sign(p1) + rho D0^2 p1 + rho D0^2 power0 + rho D1^2 p1 - rho D1^2 power1 = 0
        #     2 a p1 + rho (D0^2 + D1^2) p1 = rho D1^2 power1 - rho D0^2 power0 - b sign(p1)
        #     p1 = (rho D1^2 power1 - rho D0^2 power0 - b sign(p1)) / (2 a + rho D0^2 + rho D1^2)

        # Default is sign(num) = +1.0
        num = rho_power * (Dp2[1] * power[1] - Dp2[0] * power[0]) - data.linear_cost

        # This term is always positive, so we can pick it after choosing the sign
        denom = 2 * quadratic_cost + rho_power * (Dp2[0] + Dp2[1])
        p1 = np.divide(num, denom)

        # Finally, we project onto the box constraints
        p1 = np.clip(p1, pmin, pmax)
        p0 = -p1

        return [p0, p1], None


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
