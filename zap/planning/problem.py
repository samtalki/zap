import dataclasses
import torch
import time
import numpy as np
from copy import deepcopy

import zap.util as util
from zap.network import DispatchOutcome
from zap.layer import DispatchLayer
from zap.planning.operation_objectives import AbstractOperationObjective
from zap.planning.investment_objectives import AbstractInvestmentObjective
from .trackers import DEFAULT_TRACKERS, TRACKER_MAPS

from concurrent.futures import ThreadPoolExecutor


@dataclasses.dataclass(kw_only=True)
class GradientDescent:
    """Parameters for gradient descent."""

    step_size: float = 1e-3
    clip: float = 1e3

    def step(self, state: dict, grad: dict):
        for param in state.keys():
            grad_norm = torch.linalg.vector_norm(grad[param], ord=2)

            if grad_norm > self.clip:
                clipped_grad = (self.clip / grad_norm) * grad[param]
            else:
                clipped_grad = grad[param]

            state[param] -= self.step_size * clipped_grad.numpy()

        return state


class PlanningProblem:
    """Models long-term multi-value expansion planning."""

    def __init__(
        self,
        operation_objective: AbstractOperationObjective,
        investment_objective: AbstractInvestmentObjective,
        layer: DispatchLayer,
        lower_bounds: dict = None,
        upper_bounds: dict = None,
        regularize=1e-6,
    ):
        self.operation_objective = operation_objective
        self.investment_objective = investment_objective
        self.layer = layer
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.regularize = regularize

        if self.lower_bounds is None:
            self.lower_bounds = {
                p: getattr(layer.devices[ind], "min_" + pname, None)
                for p, (ind, pname) in self.parameter_names.items()
            }

            # Fallback: use existing device parameter value
            for p, (ind, pname) in self.parameter_names.items():
                if self.lower_bounds[p] is None:
                    self.lower_bounds[p] = getattr(layer.devices[ind], pname)

        if self.upper_bounds is None:
            self.upper_bounds = {
                p: getattr(layer.devices[ind], "max_" + pname, None)
                for p, (ind, pname) in self.parameter_names.items()
            }

            # Fallback: set to infinity
            for p, (ind, pname) in self.parameter_names.items():
                if self.upper_bounds[p] is None:
                    self.upper_bounds[p] = np.inf

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

        if requires_grad:
            la = torch
        else:
            la = np

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

        op_cost = self.operation_objective(self.torch_state, parameters=params, la=la)
        inv_cost = self.investment_objective(**torch_kwargs, la=la)

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
        t1 = time.time()
        J = self.forward(requires_grad=True, **kwargs)
        tforward = time.time() - t1
        grad = self.backward()
        t_back = time.time() - t1 - tforward
        print("Forward pass took {:.2f} seconds.".format(tforward))
        print("Backward pass took {:.2f} seconds.".format(t_back))

        return J, grad

    def solve(
        self,
        algorithm=None,
        initial_state=None,
        num_iterations=100,
        trackers=None,
        wandb=None,
        log_wandb_every=1,
        lower_bound=None,
        extra_wandb_trackers=None,
        checkpoint_every=100_000,
        checkpoint_func=lambda x: None,
    ):
        if algorithm is None:
            algorithm = GradientDescent()

        if trackers is None:
            trackers = DEFAULT_TRACKERS

        assert all([t in TRACKER_MAPS for t in trackers])
        self.start_time = time.time()
        self.lower_bound = lower_bound
        self.extra_wandb_trackers = extra_wandb_trackers

        # Setup initial state and history
        state = self.initialize_parameters(deepcopy(initial_state))
        history = self.initialize_history(trackers)

        # Initialize loop
        J, grad = self.forward_and_back(**state)
        history = self.update_history(
            history, trackers, J, grad, state, None, wandb, log_wandb_every
        )

        # Gradient descent loop
        for iteration in range(num_iterations):
            last_state = deepcopy(state)

            # Checkpoint
            if (iteration + 1) % checkpoint_every == 0:
                checkpoint_func(state, history)

            # Gradient step and project
            state = algorithm.step(state, grad)
            state = self.project(state)

            # Update loss
            J, grad = self.forward_and_back(**state)

            # Record stuff
            history = self.update_history(
                history, trackers, J, grad, state, last_state, wandb, log_wandb_every
            )

        return state, history

    def initialize_parameters(self, initial_state):
        if initial_state is None:
            return self.layer.initialize_parameters()
        else:
            return initial_state

    def initialize_history(self, trackers):
        return {k: [] for k in trackers}

    def update_history(
        self, history: dict, trackers: dict, J, grad, state, last_state, wandb, log_wandb_every
    ):
        for tracker in trackers:
            f = TRACKER_MAPS[tracker]
            f_val = f(J, grad, state, last_state, self)
            history[tracker] += [f_val]

        if wandb is not None:
            iteration = len(history[trackers[0]])

            if (iteration % log_wandb_every == 0) or (iteration == 1):
                print(f"Logging to wandb on iteration {iteration}.")

                wand_data = {tracker: history[tracker][-1] for tracker in trackers}
                wand_data["iteration"] = iteration

                # Add extra trackers
                if self.extra_wandb_trackers is not None:
                    for tracker, f in self.extra_wandb_trackers.items():
                        wand_data[tracker] = f(J, grad, state, last_state, self)

                wandb.log(wand_data)

        return history

    def project(self, state: dict):
        for param in state.keys():
            state[param] = np.clip(state[param], self.lower_bounds[param], self.upper_bounds[param])
        return state

    def get_state(self):
        return self.state

    def get_inv_cost(self):
        return self.inv_cost

    def get_op_cost(self):
        return self.op_cost

    def __add__(self, other_problem):
        return StochasticPlanningProblem([self, other_problem])

    def __mul__(self, weight):
        return StochasticPlanningProblem([self], [weight])

    def __rmul__(self, weight):
        return self.__mul__(weight)


class StochasticPlanningProblem(PlanningProblem):
    """Weighted mixture of planning problems."""

    def __init__(
        self,
        subproblems: list[PlanningProblem],
        weights: list[float] = None,
    ):
        if weights is None:
            weights = [1.0 for _ in subproblems]

        # Merge stochastic subproblems
        new_subproblems = []
        new_weights = []
        for sub, w in zip(subproblems, weights):
            if isinstance(sub, StochasticPlanningProblem):
                new_subproblems.extend(sub.subproblems)
                new_weights.extend([w * w_ for w_ in sub.weights])
            else:
                new_subproblems.append(sub)
                new_weights.append(w)

        # Drop zero weights
        subproblems = [sub for sub, w in zip(subproblems, weights) if w > 0]
        weights = [w for w in weights if w > 0]

        self.subproblems = new_subproblems
        self.weights = new_weights
        self.layer = subproblems[0].layer
        self.num_workers = 1

        # Maximum of all sub problem lower bounds
        self.lower_bounds = {
            k: np.max([sub.lower_bounds[k] for sub in self.subproblems], axis=0)
            for k in subproblems[0].lower_bounds.keys()
        }
        self.upper_bounds = {
            k: np.min([sub.upper_bounds[k] for sub in self.subproblems], axis=0)
            for k in subproblems[0].upper_bounds.keys()
        }

        assert len(self.subproblems) == len(self.weights)

    @property
    def inv_cost(self):
        return sum([w * sub.get_inv_cost() for w, sub in zip(self.weights, self.subproblems)])

    @property
    def op_cost(self):
        return sum([w * sub.get_op_cost() for w, sub in zip(self.weights, self.subproblems)])

    def initialize_workers(self, num_workers):
        self.num_workers = num_workers
        self.pool = ThreadPoolExecutor(max_workers=num_workers)
        return None

    def shutdown_workers(self):
        if self.num_workers > 1:
            self.pool.shutdown()

        self.num_workers = 1
        return None

    def forward(self, requires_grad: bool = False, **kwargs):
        if self.num_workers == 1:
            sub_costs = [sub.forward(requires_grad, **kwargs) for sub in self.subproblems]

        else:
            # Developer Note
            # Normally, multi-threading doesn't gain any performance in Python because of the GIL.
            # However, GIL is released when we call the Mosek solver.
            # This is why we can gain performance by multi-threading the forward pass.
            # The same is true for the backward pass, since the linear solver also releases the GIL.

            print(f"Threaded forward pass with {self.num_workers} workers.")
            sub_costs = self.pool.map(
                lambda sub: sub.forward(requires_grad, **kwargs), self.subproblems
            )
            sub_costs = list(sub_costs)

        return sum([w * c for w, c in zip(self.weights, sub_costs)])

    def backward(self):
        if self.num_workers == 1:
            grads = [sub.backward() for sub in self.subproblems]
        else:
            print(f"Threaded backward pass with {self.num_workers} workers.")
            grads = self.pool.map(lambda sub: sub.backward(), self.subproblems)
            grads = list(grads)

        return {k: sum([w * g[k] for w, g in zip(self.weights, grads)]) for k in grads[0].keys()}
