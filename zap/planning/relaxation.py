import cvxpy as cp

from .problem import PlanningProblem


def envelope_variable():
    pass


class RelaxedPlanningProblem:
    def __init__(self, problem: PlanningProblem):
        self.problem = problem

    def solve(self):
        # Define outer variables, constraints, and costs

        # Define primal and dual problems

        # Create full objective

        # Solve
        pass
