import torch

from zap.util import grad_or_zero
from zap.planning.operation_objectives import AbstractOperationObjective
from zap.planning.investment_objectives import AbstractInvestmentObjective
from zap.admm import ADMMLayer

from .problem_abstract import AbstractPlanningProblem


class PlanningProblemADMM(AbstractPlanningProblem):
    """Models long-term multi-value expansion planning."""

    def __init__(
        self,
        operation_objective: AbstractOperationObjective,
        investment_objective: AbstractInvestmentObjective,
        layer: ADMMLayer,
        lower_bounds: dict = None,
        upper_bounds: dict = None,
    ):
        # Call super initializer
        self.la = torch
        super().__init__(
            operation_objective, investment_objective, layer, lower_bounds, upper_bounds
        )

    def forward(self, requires_grad: bool = False, batch=None, initial_state=None, **kwargs):
        # Enable gradient tracking if needed
        for p, v in kwargs.items():
            if requires_grad:
                kwargs[p].requires_grad = True

        params = self.layer.setup_parameters(**kwargs)

        # Forward pass through dispatch layer
        # Store this for efficient backward pass
        admm_state = self.layer.forward(initial_state=initial_state, **kwargs)
        state = admm_state.as_outcome()

        op_cost = self.operation_objective(state, parameters=params, la=torch)
        inv_cost = self.investment_objective(**kwargs, la=torch)

        self.op_cost = op_cost
        self.inv_cost = inv_cost
        self.cost = op_cost + inv_cost
        self.kwargs = kwargs
        self.params = params
        self.state = state
        self.admm_state = admm_state

        return self.cost

    def backward(self):
        # Backward pass through everything!
        self.cost.backward(retain_graph=False)  # Torch backward

        # Combine gradients
        dtheta = {k: grad_or_zero(v) for k, v in self.kwargs.items()}

        return dtheta
