import torch


def replace_none(x, replacement):
    if x is None:
        return replacement
    else:
        return x


def grad_or_zero(x: torch.Tensor, to_numpy=False):
    if x is None:
        return None
    elif isinstance(x, list):
        return [grad_or_zero(xi, to_numpy=to_numpy) for xi in x]
    else:
        grad = torch.zeros_like(x) if x.grad is None else x.grad
        if to_numpy:
            return grad.numpy()
        else:
            return grad


def torchify(x, requires_grad=False):
    if x is None:
        return None
    if isinstance(x, list):
        return [torchify(xi, requires_grad=requires_grad) for xi in x]
    else:
        return torch.tensor(x, requires_grad=requires_grad)
