import torch

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
        super().__init__(
            operation_objective, investment_objective, layer, lower_bounds, upper_bounds
        )
        raise NotImplementedError

    def forward(self, requires_grad: bool = False, batch=None, **kwargs):
        # Enable gradient tracking if needed
        for p, v in kwargs.items():
            if requires_grad:
                kwargs[p].requires_grad = True

        params = self.layer.setup_parameters(**kwargs)

        # Forward pass through dispatch layer
        # Store this for efficient backward pass
        admm_state = self.layer.forward(**kwargs)
        state = admm_state.outcome_like()

        op_cost = self.operation_objective(state, parameters=params, la=torch)
        inv_cost = self.investment_objective(**kwargs, la=torch)

        self.op_cost = op_cost
        self.inv_cost = inv_cost
        self.cost = op_cost + inv_cost
        self.kwargs = kwargs
        self.params = params
        self.state = state
        self.admm_state = admm_state

        # return self.cost
        raise NotImplementedError

    def backward(self):
        # # Backward pass through operation / investment objective
        # self.cost.backward()  # Torch backward

        # # Direct component of gradients
        # dtheta_direct = {k: util.grad_or_zero(v) for k, v in self.torch_kwargs.items()}

        # # Indirect, implicitly differentiated component
        # dy = DispatchOutcome(*[util.grad_or_zero(x) for x in self.torch_state])
        # dy.ground = self.state.ground

        # # Backward pass through layer
        # dtheta_op = self.layer.backward(self.state, dy, regularize=self.regularize, **self.kwargs)

        # # Combine gradients
        # dtheta = {k: v + dtheta_op[k] for k, v in dtheta_direct.items()}

        # return dtheta
        raise NotImplementedError
