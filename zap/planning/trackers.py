import torch
import time
import numpy as np

from copy import deepcopy

LOSS = "loss"
GRAD_NORM = "grad_norm"
PROJ_GRAD_NORM = "proj_grad_norm"
PARAM = "param"
TIME = "time"


def track_loss(J, grad, state, last_state, problem):
    return J.detach().numpy()


def track_grad_norm(J, grad: dict[str, torch.Tensor], state, last_state, problem):
    """Tracks the 1-norm of the gradient."""
    return sum([torch.linalg.vector_norm(g, ord=1).item() for g in grad.values()])


def track_proj_grad_norm(J, grad, state, last_state, problem):
    """Tracks the 1-norm of the projected gradient, which is the difference between
    the current state and the previous state."""

    if last_state is None:
        return track_grad_norm(J, grad, state, last_state, problem)

    # Compute differences between states
    diffs = {k: state[k] - last_state[k] for k in state.keys()}

    return sum([np.linalg.norm(d, ord=1) for d in diffs.values()])


def track_param(J, grad, state, last_state, problem):
    return deepcopy(state)


def track_time(J, grad, state, last_state, problem):
    return time.time() - problem.start_time


TRACKER_MAPS = {
    LOSS: track_loss,
    GRAD_NORM: track_grad_norm,
    PROJ_GRAD_NORM: track_proj_grad_norm,
    PARAM: track_param,
    TIME: track_time,
}

DEFAULT_TRACKERS = [LOSS, GRAD_NORM, PROJ_GRAD_NORM, TIME]
