from zap.layer import DispatchLayer


class PlanningProblem:
    """Models long-term multi-value expansion planning."""

    def __init__(
        self,
        operation_objective,
        investment_objective,
        layer: DispatchLayer,
        lower_bounds: dict,
        upper_bounds: dict,
    ):
        self.operation_objective = operation_objective
        self.investment_objective = investment_objective
        self.layer = layer
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    @property
    def parameter_names(self):
        return self.layer.parameter_names

    @property
    def time_horizon(self):
        return self.layer.time_horizon

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, requires_grad: bool = False, **kwargs):
        # Forward pass through dispatch layer
        # Store this for efficient backward pass
        self.state = self.layer.forward(**kwargs)

        if requires_grad:
            self.torch_state = self.state.torchify()

        op_cost = self.operation_objective(self.state)
        inv_cost = self.investment_objective(**kwargs)
        return op_cost + inv_cost

    def backward(self, dJ=1.0, **kwargs):
        # Backward pass through objectives
        pass

        # Backward pass through layer
        pass
