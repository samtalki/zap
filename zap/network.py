import itertools
import cvxpy as cp

from dataclasses import dataclass
from collections import namedtuple

from zap.devices.abstract import AbstractDevice

DispatchOutcome = namedtuple(
    "DispatchOutcome",
    [
        "global_angle",
        "power",
        "angle",
        "local_variables",
        "alpha",
        "prices",
        "local_duals",
        "problem",
    ],
)


def nested_evaluate(variable):
    return [[xi.value for xi in x] if (x is not None) else None for x in variable]


def get_net_power(device: AbstractDevice, p: list[cp.Variable]) -> cp.Expression:
    return cp.sum([Ai @ pi for Ai, pi in zip(device.incidence_matrix, p)])


def match_phases(device: AbstractDevice, v, global_v):
    if v is not None:
        return [Ai.T @ global_v == vi for Ai, vi in zip(device.incidence_matrix, v)]
    else:
        return []


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
    ) -> DispatchOutcome:
        # Type checks
        assert all([d.num_nodes == self.num_nodes for d in devices])
        assert time_horizon > 0
        assert all([d.time_horizon in [0, time_horizon] for d in devices])

        if parameters is None:
            parameters = [{} for _ in devices]

        # Initialize variables
        global_angle = cp.Variable((self.num_nodes, time_horizon))
        power = [d.initialize_power(time_horizon) for d in devices]
        angle = [d.initialize_angle(time_horizon) for d in devices]
        local_variables = [d.model_local_variables(time_horizon) for d in devices]

        # Model constraints
        net_power = cp.sum([get_net_power(d, p) for d, p in zip(devices, power)])
        power_balance = net_power == 0
        reference_phase = global_angle[0] == 0
        phase_consistency = [
            match_phases(d, v, global_angle) for d, v in zip(devices, angle)
        ]
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
                [reference_phase, power_balance],
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
            alpha=reference_phase.dual_value,
            prices=-power_balance.dual_value,
            local_duals=[[lci.dual_value for lci in lc] for lc in local_constraints],
            problem=problem,
        )
