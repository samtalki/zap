import itertools
import cvxpy as cp
import numpy as np

from dataclasses import dataclass
from collections import namedtuple
from itertools import repeat

from zap.devices.abstract import AbstractDevice
from zap.devices.ground import Ground

DispatchOutcomeBase = namedtuple(
    "DispatchOutcome",
    [
        "global_angle",
        "power",
        "angle",
        "local_variables",
        "prices",
        "phase_duals",
        "local_equality_duals",
        "local_inequality_duals",
        "problem",
        "ground",
    ],
)


class DispatchOutcome(DispatchOutcomeBase):
    def _safe_cat(self, x):
        if len(x) > 0:
            return np.concatenate(x)
        else:
            return []

    def vectorize(self):
        p = self._safe_cat([np.array(p).flatten() for p in self.power])
        a = self._safe_cat([np.array(a).flatten() for a in self.angle if a is not None])
        u = self._safe_cat(
            [
                np.concatenate([ui.flatten() for ui in u])
                for u in self.local_variables
                if u is not None
            ]
        )
        mu = self._safe_cat(
            [np.array(mu).flatten() for mu in self.phase_duals if mu is not None]
        )
        lambda_eq = self._safe_cat(
            [
                self._safe_cat([lam.flatten() for lam in lambda_eq])
                for lambda_eq in self.local_equality_duals
            ]
        )
        lambda_ineq = self._safe_cat(
            [
                self._safe_cat([lam.flatten() for lam in lambda_ineq])
                for lambda_ineq in self.local_inequality_duals
            ]
        )

        return self._safe_cat(
            [
                self.global_angle.flatten(),
                p,
                a,
                u,
                self.prices.flatten(),
                mu,
                lambda_eq,
                lambda_ineq,
            ]
        )


def nested_evaluate(variable):
    return [[xi.value for xi in x] if (x is not None) else None for x in variable]


def apply_incidence(device: AbstractDevice, x):
    return [Ai @ xi for Ai, xi in zip(device.incidence_matrix, x)]


def apply_incidence_transpose(device: AbstractDevice, x):
    return [Ai.T @ xi for Ai, xi in zip(device.incidence_matrix, x)]


def get_net_power(device: AbstractDevice, p: list[cp.Variable]) -> cp.Expression:
    return cp.sum(apply_incidence(device, p))


def match_phases(device: AbstractDevice, v, global_v):
    if v is not None:
        return [Ai.T @ global_v == vi for Ai, vi in zip(device.incidence_matrix, v)]
    else:
        return []


def get_time_horizon(dispatch_outcome):
    return dispatch_outcome.global_angle.shape[1]


@dataclass
class PowerNetwork:
    """Defines the domain (nodes and settlement points) of the electrical system."""

    num_nodes: int

    def dispatch(
        self,
        devices: list[AbstractDevice],
        time_horizon=1,
        *,
        solver=cp.ECOS,
        parameters=None,
        add_ground=True,
    ) -> DispatchOutcome:
        # Add ground if necessary
        ground = None
        if add_ground:
            ground = Ground(num_nodes=self.num_nodes, terminal=np.array([0]))
            devices = devices + [ground]

        # Type checks
        assert all([d.num_nodes == self.num_nodes for d in devices])
        assert time_horizon > 0
        assert all([d.time_horizon in [0, time_horizon] for d in devices])

        if parameters is None:
            parameters = [{} for _ in devices]
        else:
            parameters += [{}]  # For the ground

        # Initialize variables
        global_angle = cp.Variable((self.num_nodes, time_horizon))
        power = [d.initialize_power(time_horizon) for d in devices]
        angle = [d.initialize_angle(time_horizon) for d in devices]
        local_variables = [d.model_local_variables(time_horizon) for d in devices]

        # Model constraints
        net_power = cp.sum([get_net_power(d, p) for d, p in zip(devices, power)])
        power_balance = net_power == 0
        phase_consistency = [
            match_phases(d, v, global_angle) for d, v in zip(devices, angle)
        ]

        local_equalities = [
            [hi == 0 for hi in d.equality_constraints(p, v, u, **param, la=cp)]
            for d, p, v, u, param in zip(
                devices, power, angle, local_variables, parameters
            )
        ]

        local_inequalities = [
            [gi <= 0 for gi in d.inequality_constraints(p, v, u, **param, la=cp)]
            for d, p, v, u, param in zip(
                devices, power, angle, local_variables, parameters
            )
        ]

        # Model objective
        costs = [
            d.operation_cost(p, v, u, **param, la=cp)
            for d, p, v, u, param in zip(
                devices, power, angle, local_variables, parameters
            )
        ]

        # Formulate and solve cvxpy problem
        objective = cp.Minimize(cp.sum(costs))
        problem = cp.Problem(
            objective,
            itertools.chain(
                [power_balance],
                *phase_consistency,
                *local_equalities,
                *local_inequalities,
            ),
        )
        problem.solve(solver=solver)

        # Evaluate variables
        power = nested_evaluate(power)
        angle = nested_evaluate(angle)
        global_angle = global_angle.value
        local_variables = nested_evaluate(local_variables)

        return DispatchOutcome(
            global_angle,
            power,
            angle,
            local_variables,
            prices=-power_balance.dual_value,
            phase_duals=[
                [pci.dual_value for pci in pc] if pc is not None else None
                for pc in phase_consistency
            ],
            local_equality_duals=[
                [lci.dual_value for lci in lc] for lc in local_equalities
            ],
            local_inequality_duals=[
                [lci.dual_value for lci in lc] for lc in local_inequalities
            ],
            problem=problem,
            ground=ground,
        )

    def kkt(self, devices, result, parameters=None):
        if result.ground is not None:
            devices = devices + [result.ground]

        if parameters is None:
            parameters = [{} for _ in devices]
        else:
            parameters += [{}]

        power = result.power
        angle = result.angle
        local_vars = result.local_variables
        lambda_eq = result.local_equality_duals
        lambda_ineq = result.local_inequality_duals

        # Local constraints - primal feasibility
        kkt_local_equalities = [
            d.equality_constraints(p, a, u, **param)
            for d, p, a, u, param in zip(devices, power, angle, local_vars, parameters)
        ]

        kkt_local_inequalities = [
            [
                np.multiply(hi, lamb_i)
                for hi, lamb_i in zip(d.inequality_constraints(p, a, u, **param), lamb)
            ]
            for d, p, a, u, param, lamb in zip(
                devices, power, angle, local_vars, parameters, lambda_ineq
            )
        ]

        # Local variables - dual feasibility
        local_grads = [
            d.lagrangian_gradients(p, a, u, lam_eq, lam_ineq, **param)
            for d, p, a, u, param, lam_eq, lam_ineq in zip(
                devices, power, angle, local_vars, parameters, lambda_eq, lambda_ineq
            )
        ]

        kkt_power = [grad[0] for grad in local_grads]
        for kp, d in zip(kkt_power, devices):
            nu_local = apply_incidence_transpose(
                d, repeat(result.prices, d.num_terminals_per_device)
            )
            for kpi, nu_local_i in zip(kp, nu_local):
                kpi -= nu_local_i

        kkt_local_angles = [grad[1] for grad in local_grads]
        for kp, d, mu in zip(kkt_local_angles, devices, result.phase_duals):
            if d.is_ac:
                for kpi, mui in zip(kp, mu):
                    kpi -= mui

        kkt_local_variables = [grad[2] for grad in local_grads]

        return DispatchOutcome(
            global_angle=self._kkt_global_angle(devices, result),
            power=kkt_power,
            angle=kkt_local_angles,
            local_variables=kkt_local_variables,
            prices=self._kkt_power_balance(devices, result),
            phase_duals=self._kkt_phase_consistency(devices, result),
            local_equality_duals=kkt_local_equalities,
            local_inequality_duals=kkt_local_inequalities,
            problem="KKT",
            ground=result.ground,
        )

    def _kkt_power_balance(self, devices, dispatch_outcome):
        net_powers = [
            get_net_power(d, p) for d, p in zip(devices, dispatch_outcome.power)
        ]
        return np.sum(net_powers, axis=0)

    def _kkt_global_angle(self, devices, dispatch_outcome):
        angle_duals = [
            np.sum(apply_incidence(d, mu), axis=0)
            for d, mu in zip(devices, dispatch_outcome.phase_duals)
            if d.is_ac
        ]
        return np.sum(angle_duals, axis=0)

    def _kkt_phase_consistency(self, devices, dispatch_outcome):
        # Compute observed global angles
        theta_terminals = [
            apply_incidence_transpose(
                d, repeat(dispatch_outcome.global_angle, d.num_terminals_per_device)
            )
            for d in devices
        ]

        # Compute phase differences
        phase_diffs = [
            np.array(theta) - np.array(a) if a is not None else None
            for theta, a in zip(theta_terminals, dispatch_outcome.angle)
        ]

        return phase_diffs

    # def kkt_jacobian_variables(self, devices, dispatch_outcome):
    #     pass

    # def kkt_jacobian_parameters(self, devices, dispatch_outcome):
    #     pass

    # def vector_jacobian_product(self, devices, dispatch_outcome, y):
    #     pass
