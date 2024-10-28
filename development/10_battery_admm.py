import marimo

__generated_with = "0.7.1"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import pandas as pd
    import scipy.sparse as sp

    import torch
    import importlib
    import pypsa
    import datetime as dt

    from copy import deepcopy
    return cp, deepcopy, dt, importlib, mo, np, pd, pypsa, sp, torch


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_theme(style="whitegrid")
    return plt, seaborn


@app.cell
def __():
    import zap
    return zap,


@app.cell
def __(pypsa):
    pn = pypsa.Network()
    pn.import_from_csv_folder("data/pypsa/western/elec_s_1000")
    return pn,


@app.cell(hide_code=True)
def __():
    DEFAULT_PYPSA_KWARGS = {
        "marginal_load_value": 500.0,
        "load_cost_perturbation": 50.0,
        "generator_cost_perturbation": 1.0,
        "cost_unit": 100.0,  # 1000.0,
        "power_unit": 1000.0,
    }
    return DEFAULT_PYPSA_KWARGS,


@app.cell(hide_code=True)
def __(DEFAULT_PYPSA_KWARGS, deepcopy, dt, pd, zap):
    def load_pypsa_network(
        pn,
        time_horizon=1,
        start_date=dt.datetime(2019, 1, 2, 0),
        exclude_batteries=False,
        **pypsa_kwargs,
    ):
        all_kwargs = deepcopy(DEFAULT_PYPSA_KWARGS)
        all_kwargs.update(pypsa_kwargs)

        dates = pd.date_range(
            start_date,
            start_date + dt.timedelta(hours=time_horizon),
            freq="1h",
            inclusive="left",
        )

        net, devices = zap.importers.load_pypsa_network(pn, dates, **all_kwargs)
        if exclude_batteries:
            devices = devices[:-1]

        return net, devices, time_horizon
    return load_pypsa_network,


@app.cell
def __():
    window = 24
    return window,


@app.cell
def __(load_pypsa_network, pn):
    net, devices, time_horizon = load_pypsa_network(pn, time_horizon=24 * 32)
    return devices, net, time_horizon


@app.cell
def __(devices, time_horizon):
    battery_cpu = devices[4]
    battery = battery_cpu.torchify(machine="cuda")
    T = time_horizon
    N = battery.num_devices

    print(N)
    return N, T, battery, battery_cpu


@app.cell
def __(N, T, devices, np):
    np.random.seed(0)

    # Load curve
    total_demand = np.sum(devices[1].min_power * devices[1].nominal_capacity, axis=0)
    z = np.random.rand(N, T)
    z = z / np.sum(z, axis=0)
    z = z * total_demand * 0.05

    # Random
    # z = np.random.randn(N, T)
    return total_demand, z


@app.cell
def __(torch, z):
    z_torch = torch.tensor(z, device="cuda", dtype=torch.float32)
    return z_torch,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Proximal Updates via CVXPY")
    return


@app.cell(hide_code=True)
def __(T, cp, np, window):
    def battery_prox_cvx(device, rho, z):
        N = device.num_devices
        pmax = device.power_capacity
        smax = np.multiply(pmax, device.duration)
        num_scenarios = T // window

        alpha = device.linear_cost
        beta = device.charge_efficiency
        gamma1 = np.multiply(device.initial_soc, smax)
        gammaT = np.multiply(device.final_soc, smax)

        # Variables
        zz = [z[:, i * window : (i + 1) * window] for i in range(num_scenarios)]
        ss = [cp.Variable((N, window + 1)) for _ in range(num_scenarios)]
        cc = [cp.Variable((N, window)) for _ in range(num_scenarios)]
        dd = [cp.Variable((N, window)) for _ in range(num_scenarios)]

        # Constraints
        eq_constraints = [
            [
                s[:, 1:] == s[:, :-1] + cp.multiply(beta, c) - d,
                s[:, 0:1] == gamma1,
                s[:, window : (window + 1)] == gammaT,
            ]
            for s, c, d in zip(ss, cc, dd)
        ]
        ineq_constraints = [
            [
                0 <= c,
                c <= pmax,
                0 <= d,
                d <= pmax,
                0 <= s,
                s <= smax,
            ]
            for s, c, d in zip(ss, cc, dd)
        ]

        # Objective function
        objective = cp.Minimize(
            sum(
                [
                    cp.sum(cp.multiply(alpha, d))
                    + (rho / 2.0) * cp.sum_squares(d - c - zi)
                    for s, c, d, zi in zip(ss, cc, dd, zz)
                ]
            )
        )

        # Solve
        problem = cp.Problem(
            objective,
            sum(eq_constraints, start=[]) + sum(ineq_constraints, start=[]),
        )
        problem.solve(solver=cp.MOSEK)

        return (
            np.hstack([c.value for c in cc]),
            np.hstack([d.value for d in dd]),
            np.hstack([s.value for s in ss]),
        )
    return battery_prox_cvx,


@app.cell
def __(battery_cpu, battery_prox_cvx, np, rho, z):
    c_cvx, d_cvx, s_cvx = battery_prox_cvx(battery_cpu, rho, z)
    p_cvx = d_cvx - c_cvx
    print(np.sum(np.abs(p_cvx)))
    return c_cvx, d_cvx, p_cvx, s_cvx


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Proximal Updates via ADMM")
    return


@app.cell
def __(torch):
    def difference_matrix(T, machine=None):
        # Should return a (T, T+1) matrix where
        # D = [-1 1 0 0]
        #     [0 -1 1 0]
        #     [0 0 -1 1]
        # for T = 3

        D1 = torch.eye(T+1, device=machine)[0:T, :]
        D2 = torch.eye(T+1, device=machine)[1:, :]

        return D2 - D1
    return difference_matrix,


@app.cell
def __(torch):
    def b_vector(device, T, machine=None):
        # TODO - support multiple costs
        alpha = device.linear_cost[0]
        return torch.vstack(
            [
                alpha * torch.ones((T, 1), device=machine),
                torch.zeros((2 * T + 1, 1), device=machine),
            ]
        )
    return b_vector,


@app.cell
def __(difference_matrix, torch):
    def C_matrix(device, T, machine=None):
        # TODO - Support multiple charge efficiencies
        beta = device.charge_efficiency[0]

        D = difference_matrix(T, machine)
        I = torch.eye(T, device=machine)

        return torch.hstack([-beta * I, I, D])
    return C_matrix,


@app.cell
def __(torch):
    def A_matrix(T, machine=None):
        I = torch.eye(T, device=machine)

        return torch.hstack([-I, I, torch.zeros(T, T+1, device=machine)])
    return A_matrix,


@app.cell
def __(A_matrix, C_matrix, torch):
    def K_matrix(device, T, rho, w, machine=None):

        A = A_matrix(T, machine)
        C = C_matrix(device, T, machine)
        I = torch.eye(3*T+1, device=machine)

        dKdx = rho * (A.T @ A) + w * I

        row1 = torch.hstack([dKdx, C.T])
        row2 = torch.hstack([C, torch.zeros(T, T, device=machine)])

        return torch.vstack([row1, row2])
    return K_matrix,


@app.cell
def __(A_matrix, b_vector, torch):
    def K_rhs_fixed(device, T, rho, z, machine=None):
        # rho * A.T @ z - b
        A = A_matrix(T, machine)
        b = b_vector(device, T, machine)
        return rho * torch.matmul(A.T, z) - b


    def K_rhs_variable(w, y, u):
        # w (y - u)
        return w * (y - u)
    return K_rhs_fixed, K_rhs_variable


@app.cell
def __():
    # difference_matrix(3)
    return


@app.cell
def __():
    # K_rhs_fixed(battery, T, rho, z_torch, machine="cuda").shape
    return


@app.cell
def __():
    # K_matrix(battery, 3, rho, 10.0, machine="cuda").shape
    return


@app.cell
def __(K_matrix, K_rhs_fixed, torch):
    def battery_prox_data(T, device, rho, z, weight=1.0, power_capacity=None):
        machine = z.device

        # Reshape z
        T_full = z.shape[1]
        num_scenarios = T_full // T
        assert T * num_scenarios == T_full

        zT = z.reshape(-1, num_scenarios, T, 1)

        N = device.num_devices
        pmax = device.power_capacity if power_capacity is None else power_capacity
        smax = torch.multiply(pmax, device.duration)

        alpha = device.linear_cost
        beta = device.charge_efficiency
        gamma1 = torch.multiply(device.initial_soc, smax)
        gammaT = torch.multiply(device.final_soc, smax)
        w = weight

        # Setup data
        K = K_matrix(device, T, rho, w, machine)
        K_inv = torch.linalg.inv(K)
        rhs = K_rhs_fixed(device, T, rho, zT, machine)
        zero_nu = torch.zeros((rhs.shape[0], rhs.shape[1], T, rhs.shape[3]), device=machine)

        ymax = torch.hstack(
            [pmax.expand(-1, 2 * T), smax.expand(-1, T + 1)]
        )
        ymin = torch.zeros(ymax.shape, device=machine)

        ymin[:, 2 * T] = gamma1[:, 0]
        ymax[:, 2 * T] = gamma1[:, 0]
        ymin[:, -1] = gammaT[:, 0]
        ymax[:, -1] = gammaT[:, 0]

        ymin = ymin.reshape(ymin.shape[0], 1, ymin.shape[1], 1)
        ymax = ymax.reshape(ymax.shape[0], 1, ymax.shape[1], 1)

        return (
            N,
            num_scenarios,
            alpha,
            beta,
            pmax,
            smax,
            gamma1,
            gammaT,
            K_inv,
            rhs,
            zero_nu,
            ymin,
            ymax,
        )
    return battery_prox_data,


@app.cell(hide_code=True)
def __():
    # _z_tensor = z_torch.reshape(-1, 2, 24)

    # assert _z_tensor[:, 0, :].shape == z_torch[:, :24].shape
    # assert torch.all(_z_tensor[:, 0, :] == z_torch[:, :24])

    # _z_tensor.transpose(0, -1).shape
    return


@app.cell
def __(
    admm_weight,
    battery,
    battery_prox_data,
    power_capacity,
    rho,
    window,
    z_torch,
):
    data = battery_prox_data(window, battery, rho, z_torch, admm_weight, power_capacity)
    return data,


@app.cell(hide_code=True)
def __(battery_prox_inner, torch):
    def battery_prox_admm(
        T,
        rho,
        z,
        num_iterations=10,
        weight=1.0,
        power_capacity=None,
        data=None,
        over_relaxation=1.0,
    ):
        machine = z.device
        (
            N,
            num_scenarios,
            alpha,
            beta,
            pmax,
            smax,
            gamma1,
            gammaT,
            K_inv,
            rhs,
            zero_nu,
            ymin,
            ymax,
        ) = data
        zT = z.reshape(-1, num_scenarios, T, 1)


        # Initialize
        # Dimensions are transposed of actual battery dimensions
        x = torch.zeros((N, num_scenarios, 3 * T + 1, 1), device=machine)
        y = torch.zeros(x.shape, device=machine)
        u = torch.zeros(x.shape, device=machine)

        y[:, :, :T, :] = torch.relu(-zT)
        y[:, :, T : (2 * T), :] = torch.relu(zT)
        y[:, :, (2 * T) :, :] = gamma1.reshape(-1, 1, 1, 1)
        # y[(2*T+1):, :] += -z.T

        # Solve ADMM
        for iter in range(num_iterations):
            x, y, u = battery_prox_inner(
                x, y, u, rhs, zero_nu, K_inv, ymin, ymax, weight, over_relaxation
            )

        # Use y, not x, because it's guaranteed to satsify power limits
        # (but not charge limits... hmmm)
        c = x[:, :, :T, 0].reshape(N, -1)
        d = x[:, :, T : (2 * T), 0].reshape(N, -1)
        s = x[:, :, (2 * T) :, 0].reshape(N, -1)

        return c, d, s
    return battery_prox_admm,


@app.cell(hide_code=True)
def __(torch):
    @torch.jit.script
    def battery_prox_inner(
        x, y, u, rhs, zero_nu, K_inv, ymin, ymax, w: float, alpha: float = 1.0
    ):
        # x update
        rhs_var = rhs + w * (y - u)
        full_rhs = torch.cat([rhs_var, zero_nu], dim=2)
        x = (K_inv @ full_rhs)[:, :, : x.shape[2], :]

        # over relaxation step
        xp = alpha * x + (1 - alpha) * y

        # y update
        y = torch.clip(xp + u, min=ymin, max=ymax)

        # u update
        u += xp - y

        return x, y, u
    return battery_prox_inner,


@app.cell
def __(battery):
    power_capacity = battery.power_capacity.clone()
    power_capacity.requires_grad = False
    return power_capacity,


@app.cell
def __():
    import time
    return time,


@app.cell
def __(
    admm_weight,
    battery_prox_admm,
    data,
    num_admm_iter,
    over_relaxation_weight,
    power_capacity,
    rho,
    time,
    torch,
    window,
    z_torch,
):
    _start_time = time.time()

    c_ad, d_ad, s_ad = battery_prox_admm(
        window, rho, z_torch, num_iterations=num_admm_iter, weight=admm_weight,
        power_capacity=power_capacity,
        data=data,
        over_relaxation=over_relaxation_weight
    )

    if power_capacity.requires_grad:
        c_ad.backward(torch.rand(c_ad.shape, device="cuda"))

    _runtime = 1000 * (time.time() - _start_time)
    print(f"{_runtime:.2f} ms")
    return c_ad, d_ad, s_ad


@app.cell
def __(c_ad, d_ad, s_ad):
    c_admm = c_ad.detach().to("cpu").numpy()
    d_admm = d_ad.detach().to("cpu").numpy()
    s_admm = s_ad.detach().to("cpu").numpy()
    p_admm = d_admm - c_admm
    return c_admm, d_admm, p_admm, s_admm


@app.cell
def __():
    rho = 2.0
    return rho,


@app.cell
def __(rho):
    num_admm_iter = 30
    admm_weight = rho / 2.0
    over_relaxation_weight = 1.8
    return admm_weight, num_admm_iter, over_relaxation_weight


@app.cell(hide_code=True)
def __(
    admm_weight,
    battery,
    num_admm_iter,
    over_relaxation_weight,
    rho,
    window,
    z_torch,
):
    battery.has_changed = True

    p_admm_2, _ = battery.admm_prox_update(
        rho,
        None,
        [z_torch],
        None,
        inner_weight=admm_weight / rho,
        inner_over_relaxation=over_relaxation_weight,
        inner_iterations=num_admm_iter,
        window=window,
    )

    p_admm_2 = p_admm_2[0].detach().cpu().numpy()
    return p_admm_2,


@app.cell
def __(np, p_admm, p_admm_2):
    np.linalg.norm((p_admm - p_admm_2).ravel(), 1)
    return


@app.cell
def __():
    debug_index = 15
    return debug_index,


@app.cell
def __(T, window):
    num_scenarios = T // window
    return num_scenarios,


@app.cell(hide_code=True)
def __(
    T,
    battery_cpu,
    debug_index,
    np,
    num_scenarios,
    p_admm,
    p_admm_2,
    p_cvx,
    plt,
    s_admm,
    s_cvx,
):
    _ind = debug_index

    print("Absolute Error:", np.sum(np.abs(p_admm - p_cvx)))
    print("Relative Error:", np.sum(np.abs(p_admm - p_cvx)) / np.sum(np.abs(p_cvx)))

    print("\nV2 Absolute Error:", np.sum(np.abs(p_admm_2 - p_cvx)))
    print("V2 Relative Error:", np.sum(np.abs(p_admm_2 - p_cvx)) / np.sum(np.abs(p_cvx)))

    _smax = battery_cpu.power_capacity[_ind] * battery_cpu.duration[_ind]
    _s0 = battery_cpu.initial_soc[_ind] * _smax

    plt.figure(figsize=(6, 2))
    plt.step(range(T+num_scenarios), s_cvx[_ind, :] / _smax, label="cvxpy", where="mid")
    plt.step(range(T+num_scenarios), s_admm[_ind, :] / _smax, label="admm", ls="dashed", where="mid")
    plt.ylim(-0.1, 1.1)
    plt.legend(loc="upper right")
    return


@app.cell(hide_code=True)
def __(
    battery_cpu,
    c_admm,
    c_cvx,
    d_admm,
    d_cvx,
    debug_index,
    np,
    plt,
    s_admm,
    s_cvx,
    window,
):
    _ind = debug_index
    _s, _c, _d = s_cvx, c_cvx, d_cvx
    _s, _c, _d = s_admm, c_admm, d_admm

    smax = battery_cpu.power_capacity[_ind] * battery_cpu.duration[_ind]
    s0 = battery_cpu.initial_soc[_ind] * smax

    plt.figure(figsize=(6, 2))
    plt.step(np.arange(window + 1), _s[_ind, :window+1], label="soc", where="mid")
    plt.step(
        np.arange(1, window+1),
        _c[_ind, :window] - _d[_ind, :window] + _s[_ind, :window],
        label="power",
        color="black",
        where="mid",
        ls="dotted",
        # bottom=_s[_ind, :T],
    )
    plt.ylim(-0.1, 1.1*smax)

    plt.show()
    return s0, smax


if __name__ == "__main__":
    app.run()
