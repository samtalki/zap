import numpy as np
import torch

from zap.devices.abstract import AbstractDevice


def infer_machine():
    return "cuda" if torch.cuda.is_available() else "cpu"


def scatter_sum(num_rows, index, source):
    num_cols = source.shape[1]
    result = torch.zeros((num_rows, num_cols), device=source.device, dtype=source.dtype)
    return result.scatter_add(0, index, source)


def gather_sum(index, source):
    return torch.gather(source, 0, index)


def apply_incidence(device: AbstractDevice, x_list: list[torch.Tensor]):
    # machine = x_list[0].device
    # if machine.type == "cpu":
    #     return apply_incidence_cpu(device, x_list)
    # else:
    #     return apply_incidence_gpu(device, x_list)
    return apply_incidence_gpu(device, x_list)


def apply_incidence_transpose(device: AbstractDevice, x: torch.Tensor):
    # machine = x.device
    # if machine.type == "cpu":
    #     return apply_incidence_transpose_cpu(device, x)
    # else:
    #     return apply_incidence_transpose_gpu(device, x)
    return apply_incidence_transpose_gpu(device, x)


def apply_incidence_cpu(device: AbstractDevice, x_list: list[torch.Tensor]):
    return [A_dt @ x_dt for A_dt, x_dt in zip(device.incidence_matrix, x_list)]


def apply_incidence_gpu(device: AbstractDevice, x_list: list[torch.Tensor]):
    n, T = device.num_nodes, x_list[0].shape[1]
    machine = x_list[0].device
    return [
        scatter_sum(n, tau, x_dt) for tau, x_dt in zip(device.torch_terminals(T, machine), x_list)
    ]


def apply_incidence_transpose_cpu(device: AbstractDevice, x: torch.Tensor):
    return [A_dt.T @ x for A_dt in device.incidence_matrix]


def apply_incidence_transpose_gpu(device: AbstractDevice, x: torch.Tensor):
    T = x.shape[1]
    machine = x.device
    return [gather_sum(tau, x) for tau in device.torch_terminals(T, machine)]


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


def get_num_terminals(net, devices, only_ac=False, machine=None):
    if machine is None:
        machine = infer_machine()

    terminal_counts = np.zeros(net.num_nodes)
    for d in devices:
        if only_ac and (not d.is_ac):
            continue

        values, counts = np.unique(d.terminals, return_counts=True)
        for t, c in zip(values, counts):
            terminal_counts[t] += c

    return torch.tensor(np.expand_dims(terminal_counts, 1), device=machine)


def get_nodal_average(
    powers,
    net,
    devices,
    time_horizon,
    num_terminals=None,
    only_ac=False,
    check_connections=True,
    tol=1e-8,
    machine=None,
):
    if machine is None:
        machine = infer_machine()

    tol = torch.tensor(tol, device=machine)
    if num_terminals is None:
        num_terminals = get_num_terminals(net, devices, only_ac=only_ac, machine=machine)

    if check_connections:
        assert torch.all(num_terminals > 0)
    else:
        num_terminals = torch.maximum(num_terminals, tol)

    average_x = torch.zeros((net.num_nodes, time_horizon), device=machine)

    for dev, x_dev in zip(devices, powers):
        if x_dev is None:
            continue

        A_x = apply_incidence(dev, x_dev)
        for A_x_dt in A_x:
            average_x += A_x_dt

        # for A_dt, x_dt in zip(dev.incidence_matrix, x_dev):
        #     average_x += A_dt @ x_dt

    return torch.divide(average_x, num_terminals)


def get_terminal_residual(angles, average_angle, devices):
    AT_theta = [
        None if a_dev is None else apply_incidence_transpose(dev, average_angle)
        for a_dev, dev in zip(angles, devices)
    ]

    residuals = [
        None
        if a_dev is None
        else [a_dt - AT_theta_dt for a_dt, AT_theta_dt in zip(a_dev, AT_theta_dev)]
        for a_dev, AT_theta_dev in zip(angles, AT_theta)
    ]

    return residuals


def dc_average(x, net, devices, time_horizon, num_terminals, machine="cpu"):
    return get_nodal_average(
        x, net, devices, time_horizon, num_terminals, only_ac=False, machine=machine
    )


def ac_average(x, net, devices, time_horizon, num_ac_terminals, machine="cpu"):
    return get_nodal_average(
        x,
        net,
        devices,
        time_horizon,
        num_ac_terminals,
        only_ac=True,
        check_connections=False,
        machine=machine,
    )
