import torch
import numpy as np
import cvxpy as cp


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
    if isinstance(x, torch.Tensor):
        return x
    elif x is None:
        return None
    elif isinstance(x, list):
        return [torchify(xi, requires_grad=requires_grad) for xi in x]
    else:
        return torch.tensor(x, requires_grad=requires_grad)


def torch_sparse(A):
    if isinstance(A, list):
        return [torch_sparse(Ai) for Ai in A]

    A = A.tocoo()
    values = A.data
    indices = np.vstack((A.row, A.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    return torch.sparse_coo_tensor(i, v, torch.Size(A.shape), dtype=torch.float64)


def choose_base_modeler(la):
    if la == torch:
        return torch
    elif la in [np, cp]:
        return np
    else:
        raise ValueError(f"Unknown la: {la}")
