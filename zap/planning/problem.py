import dataclasses
import numpy as np

import zap.util as util
from zap.network import DispatchOutcome
from zap.layer import DispatchLayer
from zap.planning.operation_objectives import AbstractOperationObjective


@dataclasses.dataclass(kw_only=True)
class GradientDescent:
    """Parameters for gradient descent."""

    step_size: float = 1e-3

    def step(self, state: dict, grad: dict):
        for param in state.keys():
            state[param] -= self.step_size * grad[param].numpy()

        return state


class PlanningProblem:
    """Models long-term multi-value expansion planning."""

    def __init__(
        self,
        operation_objective: AbstractOperationObjective,
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
        torch_kwargs = {}
        for p, v in kwargs.items():
            if requires_grad:
                torch_kwargs[p] = util.torchify(v, requires_grad=True)
            else:
                torch_kwargs[p] = v

        params = self.layer.setup_parameters(**torch_kwargs)

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
        inv_cost = self.investment_objective(**torch_kwargs, use_torch=requires_grad)

        self.op_cost = op_cost
        self.inv_cost = inv_cost
        self.cost = op_cost + inv_cost

        self.kwargs = kwargs
        self.torch_kwargs = torch_kwargs
        self.params = params

        return self.cost

    def backward(self):
        # Backward pass through operation / investment objective
        self.cost.backward()  # Torch backward

        # Direct component of gradients
        dtheta_direct = {k: util.grad_or_zero(v) for k, v in self.torch_kwargs.items()}

        # Indirect, implicitly differentiated component
        dy = DispatchOutcome(*[util.grad_or_zero(x) for x in self.torch_state])
        dy.ground = self.state.ground

        # Backward pass through layer
        dtheta_op = self.layer.backward(self.state, dy, regularize=self.regularize, **self.kwargs)

        # Combine gradients
        dtheta = {k: v + dtheta_op[k] for k, v in dtheta_direct.items()}

        return dtheta

    def forward_and_back(self, **kwargs):
        J = self.forward(requires_grad=True, **kwargs)
        grad = self.backward()

        return J, grad

    def solve(self, algorithm=None, initial_state=None, num_iterations=100):
        if algorithm is None:
            algorithm = GradientDescent()

        # Setup initial state and history
        state = self.initialize_parameters(initial_state)
        history = {
            "loss": [],
        }  # TODO(later) Add other trackers

        # Initialize loop
        J, grad = self.forward_and_back(**state)
        history = self.update_history(history, J, grad, state)

        # Gradient descent loop
        for iteration in range(num_iterations):
            # Gradient step and project
            state = algorithm.step(state, grad)
            state = self.project(state)

            # Update loss
            J, grad = self.forward_and_back(**state)

            # Record stuff
            history = self.update_history(history, J, grad, state)

        return state, history

    def initialize_parameters(self, initial_state):
        if initial_state is None:
            return self.layer.initialize_parameters()
        else:
            return initial_state

    def update_history(self, history: dict, J, grad, state):
        history["loss"] += [J]
        return history

    def project(self, state: dict):
        for param in state.keys():
            state[param] = np.clip(state[param], self.lower_bounds[param], self.upper_bounds[param])
        return state
