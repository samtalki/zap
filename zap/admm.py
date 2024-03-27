import numpy as np


def get_num_terminals(net, devices, only_ac=False):
    terminal_counts = np.zeros(net.num_nodes)
    for d in devices:
        if only_ac and (not d.is_ac):
            continue

        values, counts = np.unique(d.terminals, return_counts=True)
        for t, c in zip(values, counts):
            terminal_counts[t] += c

    return np.expand_dims(terminal_counts, 1)


def get_nodal_average(
    powers, net, devices, time_horizon, num_terminals=None, only_ac=False, check_connections=True
):
    if num_terminals is None:
        num_terminals = get_num_terminals(net, devices, only_ac=only_ac)

    if check_connections:
        assert np.all(num_terminals > 0)
    else:
        num_terminals = np.maximum(num_terminals, 1e-8)

    average_x = np.zeros((net.num_nodes, time_horizon))

    for dev, x_dev in zip(devices, powers):
        if x_dev is None:
            continue
        for A_dt, x_dt in zip(dev.incidence_matrix, x_dev):
            average_x += A_dt @ x_dt

    return np.divide(average_x, num_terminals)


def get_terminal_residual(angles, average_angle, net, devices, time_horizon):
    residuals = [
        None if x_dev is None else [np.zeros_like(x_dt) for x_dt in x_dev] for x_dev in angles
    ]
    for dev, r_dev, a_dev in zip(devices, residuals, angles):
        if r_dev is None:
            continue
        for A_dt, r_dt, a_dt in zip(dev.incidence_matrix, r_dev, a_dev):
            r_dt += a_dt - A_dt.T @ average_angle

    return residuals
