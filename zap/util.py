import torch


def replace_none(x, replacement):
    if x is None:
        return replacement
    else:
        return x


def grad_or_zero(x: torch.Tensor):
    return torch.zeros_like(x) if x.grad is None else x.grad
