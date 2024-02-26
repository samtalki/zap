import itertools
import cvxpy as cp

from dataclasses import dataclass
from collections import namedtuple

from zap.devices.abstract import AbstractDevice

DispatchOutcome = namedtuple(
    "DispatchOutcome",
    ["power", "angle", "global_angle", "local_variables", "problem"],
)


def nested_evaluate(variable):
    return [[xi.value for xi in x] if (x is not None) else None for x in variable]


def get_net_power(device, p):
    return cp.sum([Ai @ pi for Ai, pi in zip(device.incidence_matrix, p)])


def match_phases(device, v, global_v):
    if v is not None:
        return [Ai.T @ global_v == vi for Ai, vi in zip(device.incidence_matrix, v)]
    else:
        return []


@dataclass
class PowerNetwork:
    """Defines the domain (nodes and settlement points) of the electrical system."""

    num_nodes: int

    def dispatch(self, devices: list[AbstractDevice], solver=cp.ECOS):
        assert all([d.num_nodes == self.num_nodes for d in devices])

        # Initialize variables
        global_angle = cp.Variable(self.num_nodes)
        power = [d.initialize_power() for d in devices]
        angle = [d.initialize_angle() for d in devices]
        local_variables = [d.model_local_variables() for d in devices]

        # Model constraints
        net_power = cp.sum([get_net_power(d, p) for d, p in zip(devices, power)])
        phase_consistency = [
            match_phases(d, v, global_angle) for d, v in zip(devices, angle)
        ]
        local_constraints = [
            d.model_local_constraints(p, v, u)
            for d, p, v, u in zip(devices, power, angle, local_variables)
        ]

        # Model objective
        costs = [
            d.model_cost(p, v, u)
            for d, p, v, u in zip(devices, power, angle, local_variables)
        ]

        # Formulate cvxpy problem
        objective = cp.Minimize(cp.sum(costs))
        problem = cp.Problem(
            objective,
            itertools.chain(
                [global_angle[0] == 0, net_power == 0],
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

        return DispatchOutcome(power, angle, global_angle, local_variables, problem)
