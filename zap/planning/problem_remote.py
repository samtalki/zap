import ray
from zap.planning.problem import PlanningProblem


@ray.remote
class RemotePlanningProblem(PlanningProblem):
    def __init__(self, problem: PlanningProblem):
        self.operation_objective = problem.operation_objective
        self.investment_objective = problem.investment_objective
        self.layer = problem.layer
        self.lower_bounds = problem.lower_bounds
        self.upper_bounds = problem.upper_bounds
        self.regularize = problem.regularize
