import cvxpy as cp
import numpy as np

from .problem import PlanningProblem


def envelope_variable():
    pass


class RelaxedPlanningProblem:
    def __init__(
        self, problem: PlanningProblem, inf_value=100.0, solver=cp.MOSEK, solver_kwargs={}
    ):
        self.problem = problem
        self.inf_value = inf_value
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def solve(self):
        # Define outer variables, constraints, and costs
        network_parameters = {
            p: cp.Variable(lower.shape) for p, lower in self.problem.lower_bounds.items()
        }

        lower_bounds = []
        upper_bounds = []

        for p in sorted(network_parameters.keys()):
            lower = self.problem.lower_bounds[p]
            upper = self.problem.upper_bounds[p]

            # Replace infs with inf_value times max value
            inf_param = self.inf_value * np.max(lower)
            upper = np.where(upper == np.inf, inf_param, upper)

            lower_bounds.append(network_parameters[p] >= lower)
            upper_bounds.append(network_parameters[p] <= upper)

        investment_objective = self.problem.investment_objective(la=cp, **network_parameters)

        # Define primal and dual problems

        # Define strong duality coupling constraint

        # Define operation objective

        # Create full problem
        problem = cp.Problem(
            cp.Minimize(investment_objective),
            lower_bounds + upper_bounds,
        )
        problem.solve(solver=self.solver, **self.solver_kwargs)

        # Solve
        return {
            "network_parameters": network_parameters,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "investment_objective": investment_objective,
            "problem": problem,
        }
