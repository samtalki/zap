from .problem_abstract import AbstractPlanningProblem
from .problem_cvx import PlanningProblemCVX
from .problem_admm import PlanningProblemADMM

from zap.admm import ADMMLayer


def PlanningProblem(
    operation_objective, investment_objective, layer, lower_bounds=None, upper_bounds=None, **kwargs
) -> AbstractPlanningProblem:
    if isinstance(layer, ADMMLayer):
        return PlanningProblemADMM(
            operation_objective,
            investment_objective,
            layer,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            **kwargs,
        )
    else:
        return PlanningProblemCVX(
            operation_objective,
            investment_objective,
            layer,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            **kwargs,
        )
