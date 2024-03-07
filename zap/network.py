import itertools
import cvxpy as cp
import numpy as np

from dataclasses import dataclass
from collections import namedtuple
from itertools import repeat

from zap.devices.abstract import AbstractDevice
from zap.devices.ground import Ground

DispatchOutcome = namedtuple(
    "DispatchOutcome",
    [
        "global_angle",
        "power",
        "angle",
        "local_variables",
        "prices",
        "phase_duals",
        "local_duals",
        "problem",
        "ground",
    ],
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

        # Add local constraints
        local_constraints = [
            d.model_local_constraints(p, v, u, **param)
            for d, p, v, u, param in zip(
                devices, power, angle, local_variables, parameters
            )
        ]

        # Model objective
        costs = [
            d.model_cost(p, v, u, **param)
            for d, p, v, u, param in zip(
                devices, power, angle, local_variables, parameters
            )
        ]

        # Formulate cvxpy problem
        objective = cp.Minimize(cp.sum(costs))
        problem = cp.Problem(
            objective,
            itertools.chain(
                [power_balance],
                *phase_consistency,
                *local_constraints,
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
            local_duals=[[lci.dual_value for lci in lc] for lc in local_constraints],
            problem=problem,
            ground=ground,
        )

    def kkt(self, devices, dispatch_outcome):
        if dispatch_outcome.ground is not None:
            devices = devices + [dispatch_outcome.ground]

        # Local constraints - primal feasibility
        # kkt_local_equalities = None
        # kkt_local_inequalities = None

        # Local variables - dual feasibility
        # kkt_local_variables = None
        # kkt_local_angles = None
        # kkt_local_powers = None

        return DispatchOutcome(
            global_angle=self._kkt_global_angle(devices, dispatch_outcome),
            power=None,
            angle=None,
            local_variables=None,
            prices=self._kkt_power_balance(devices, dispatch_outcome),
            phase_duals=self._kkt_phase_consistency(devices, dispatch_outcome),
            local_duals=None,
            problem="KKT",
            ground=dispatch_outcome.ground,
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
