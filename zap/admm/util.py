import numpy as np
import torch


def nested_map(f, *args, none_value=None):
    return [
        (none_value if arg_dev[0] is None else [f(*arg_dt) for arg_dt in zip(*arg_dev)])
        for arg_dev in zip(*args)
    ]


def nested_add(x1, x2, alpha=None):
    if alpha is None:
        return nested_map(lambda x, y: x + y, x1, x2)
    else:
        return nested_map(lambda x, y: alpha * (x + y), x1, x2)


def nested_subtract(x1, x2, alpha=None):
    if alpha is None:
        return nested_map(lambda x, y: x - y, x1, x2)
    else:
        return nested_map(lambda x, y: alpha * (x - y), x1, x2)


def nested_norm(data, p=None):
    mini_norms = [
        (
            torch.tensor([0.0])
            if x_dev is None
            else torch.tensor([torch.linalg.norm(x.ravel(), p) for x in x_dev])
        )
        for x_dev in data
    ]
    return torch.linalg.norm(torch.concatenate(mini_norms), p)


def get_discrep(power1, power2):
    discreps = [
        (
            [0.0]
            if p_admm is None
            else [torch.linalg.norm(p1 - p2, 1) for p1, p2 in zip(p_admm, p_cvx)]
        )
        for p_admm, p_cvx in zip(power1, power2)
    ]
    return torch.sum(torch.tensor(discreps))


def get_num_terminals(net, devices, only_ac=False):
    terminal_counts = np.zeros(net.num_nodes)
    for d in devices:
        if only_ac and (not d.is_ac):
            continue

        values, counts = np.unique(d.terminals, return_counts=True)
        for t, c in zip(values, counts):
            terminal_counts[t] += c

    return torch.tensor(np.expand_dims(terminal_counts, 1))


def get_nodal_average(
    powers,
    net,
    devices,
    time_horizon,
    num_terminals=None,
    only_ac=False,
    check_connections=True,
    tol=torch.tensor(1e-8),
):
    if num_terminals is None:
        num_terminals = get_num_terminals(net, devices, only_ac=only_ac)

    if check_connections:
        assert torch.all(num_terminals > 0)
    else:
        num_terminals = torch.maximum(num_terminals, tol)

    average_x = torch.zeros((net.num_nodes, time_horizon))

    for dev, x_dev in zip(devices, powers):
        if x_dev is None:
            continue
        for A_dt, x_dt in zip(dev.incidence_matrix, x_dev):
            average_x += A_dt @ x_dt

    return torch.divide(average_x, num_terminals)


def get_terminal_residual(angles, average_angle, devices):
    residuals = [
        None
        if a_dev is None
        else [a_dt - A_dt.T @ average_angle for a_dt, A_dt in zip(a_dev, dev.incidence_matrix)]
        for a_dev, dev in zip(angles, devices)
    ]

    return residuals


def dc_average(x, net, devices, time_horizon, num_terminals):
    return get_nodal_average(x, net, devices, time_horizon, num_terminals)


def ac_average(x, net, devices, time_horizon, num_ac_terminals):
    return get_nodal_average(
        x, net, devices, time_horizon, num_ac_terminals, only_ac=True, check_connections=False
    )
