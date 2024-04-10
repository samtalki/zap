import zap.util as util

from zap.network import DispatchOutcome
from zap.layer import DispatchLayer
from zap.planning.objectives import AbstractObjective


class PlanningProblem:
    """Models long-term multi-value expansion planning."""

    def __init__(
        self,
        operation_objective: AbstractObjective,
        investment_objective,
        layer: DispatchLayer,
        lower_bounds: dict,
        upper_bounds: dict,
        regularize=1e-6,
    ):
        self.operation_objective = operation_objective
        self.investment_objective = investment_objective
        self.layer = layer
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.regularize = regularize

    @property
    def parameter_names(self):
        return self.layer.parameter_names

    @property
    def time_horizon(self):
        return self.layer.time_horizon

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, requires_grad: bool = False, **kwargs):
        params = self.layer.setup_parameters(**kwargs)

        # Forward pass through dispatch layer
        # Store this for efficient backward pass
        self.state = self.layer.forward(**kwargs)

        if requires_grad:
            self.torch_state = self.state.torchify(requires_grad=True)
        else:
            self.torch_state = self.state

        op_cost = self.operation_objective(
            self.torch_state, parameters=params, use_torch=requires_grad
        )
        inv_cost = self.investment_objective(**kwargs)

        self.cost = op_cost  # + inv_cost
        return self.cost

    def backward(self, **kwargs):
        # Backward pass through objectives
        self.cost.backward()  # Torch backward
        dy = DispatchOutcome(*[util.grad_or_zero(x) for x in self.torch_state])
        dy.ground = self.state.ground

        # Backward pass through layer
        dtheta = self.layer.backward(self.state, dy, regularize=self.regularize, **kwargs)

        return dtheta

    def forward_and_back(self, **kwargs):
        J = self.forward(requires_grad=True, **kwargs)
        grad = self.backward(**kwargs)

        return J, grad
