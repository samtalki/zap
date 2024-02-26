import itertools
import cvxpy as cp

from dataclasses import dataclass
from collections import namedtuple

DispatchOutcome = namedtuple(
    "DispatchOutcome",
    ["power", "angle", "local_variables", "problem"],
)


def nested_evaluate(variable):
    return [[xi.value for xi in x] if (x is not None) else None for x in variable]


@dataclass
class PowerNetwork:
    """Defines the domain (nodes and settlement points) of the electrical system."""

    num_nodes: int

    def _initialize_power(self, devices):
        return [
            [cp.Variable(d.num_devices) for _ in range(d.num_terminals_per_device)]
            for d in devices
        ]

    def _initialize_angle(self, devices):
        return [
            [cp.Variable(d.num_devices) for _ in range(d.num_terminals_per_device)]
            if d.is_ac
            else None
            for d in devices
        ]

    def dispatch(self, devices, solver=cp.ECOS):
        # Initialize variables
        power = self._initialize_power(devices)
        angle = self._initialize_angle(devices)
        local_variables = [d.model_local_variables() for d in devices]

        # Model constraints
        power_balance = None  # TODO
        phase_consistency = None  # TODO
        local_constraints = [
            d.model_local_constraints(p, v, u)
            for d, p, v, u in zip(devices, power, angle, local_variables)
        ]

        # Model objective
        costs = [
            d.model_cost(p, v, u)
            for d, p, v, u in zip(devices, power, angle, local_variables)
        ]

        objective = cp.Minimize(cp.sum(costs))

        # Formulate cvxpy problem
        problem = cp.Problem(objective, itertools.chain(*local_constraints))
        problem.solve(solver=solver)

        # Evaluate variables
        power = nested_evaluate(power)
        angle = nested_evaluate(angle)
        local_variables = nested_evaluate(local_variables)

        return DispatchOutcome(power, angle, local_variables, problem)
