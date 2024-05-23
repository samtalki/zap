import marimo

__generated_with = "0.4.3"
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
    import zap
    from zap import DispatchLayer
    return DispatchLayer, zap


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn
    seaborn.set_theme()
    return plt, seaborn


@app.cell
def __(mo):
    mo.md("## Network")
    return


@app.cell
def __():
    DEFAULT_PYPSA_KWARGS = {
        "marginal_load_value": 500.0,
        "load_cost_perturbation": 50.0,
        "generator_cost_perturbation": 1.0,
        "cost_unit": 100.0,  # 1000.0,
        "power_unit": 1000.0,
    }
    return DEFAULT_PYPSA_KWARGS,


@app.cell
def __(pypsa):
    pn = pypsa.Network()
    pn.import_from_csv_folder("data/pypsa/western/elec_s_100")
    return pn,


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
        print(all_kwargs)

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
def __(load_pypsa_network, pn):
    net, devices, time_horizon = load_pypsa_network(pn, time_horizon=24 * 1)

    for _d in devices:
        print(type(_d))
    return devices, net, time_horizon


@app.cell
def __():
    # result = net.dispatch(
    #     devices,
    #     time_horizon,
    #     solver=cp.MOSEK,
    #     add_ground=False,
    #     solver_kwargs={"verbose": False}
    # )
    # result.problem.value
    return


@app.cell
def __():
    # _x = result.torchify(machine="cuda")

    # # Compute global power / phase imbalance
    # average_power = get_nodal_average(_x.power, net, devices, time_horizon)
    # average_angle = get_nodal_average(
    #     _x.angle, net, devices, time_horizon, only_ac=True
    # )
    # global_phase_imbalance = average_angle - _x.global_angle

    # print(f"Power Imbalance: {torch.linalg.norm(average_power, 1)}")
    # print(f"Global Phase Imbalance: {torch.linalg.norm(global_phase_imbalance, 1)}")

    # # Compute local phase imbalance
    # phase_residual = get_terminal_residual(_x.angle, average_angle, devices)

    # print(f"Local Phase Imbalance: {nested_norm(phase_residual)}")
    return


@app.cell
def __(mo):
    mo.md("## Solve with CVXPY")
    return


@app.cell
def __(deepcopy, devices, net, np, zap):
    simple_devices = deepcopy(devices[:3])
    use_ac = True

    # Add AC or DC lines
    if use_ac:
        simple_devices += [deepcopy(devices[3])]
    else:
        simple_devices += [
            deepcopy(
                zap.DCLine(
                    num_nodes=devices[3].num_nodes,
                    source_terminal=devices[3].source_terminal,
                    sink_terminal=devices[3].sink_terminal,
                    capacity=devices[3].capacity,
                    nominal_capacity=devices[3].nominal_capacity,
                    linear_cost=devices[3].linear_cost,
                )
            )
        ]

    _ground = zap.Ground(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        voltage=np.array([0.0]),
    )
    simple_devices += [_ground]

    for _d in simple_devices:
        print(type(_d))
    return simple_devices, use_ac


@app.cell
def __(cp, nested_norm, net, simple_devices, time_horizon):
    # Dispatch
    simple_result = net.dispatch(
        simple_devices,
        time_horizon,
        solver=cp.MOSEK,
        add_ground=False,
    )

    print(simple_result.problem.solver_stats)
    print(nested_norm(simple_result.torchify().angle))
    print(nested_norm(simple_result.torchify().power))
    return simple_result,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## ADMM")
    return


@app.cell
def __():
    from zap.admm import ADMMSolver, WeightedADMMSolver
    return ADMMSolver, WeightedADMMSolver


@app.cell
def __():
    from zap.admm import nested_map, nested_norm, get_discrep
    return get_discrep, nested_map, nested_norm


@app.cell
def __():
    from zap.admm import get_nodal_average, get_num_terminals, get_terminal_residual
    return get_nodal_average, get_num_terminals, get_terminal_residual


@app.cell
def __():
    from zap.admm import ac_average, dc_average
    return ac_average, dc_average


@app.cell(hide_code=True)
def __(mo):
    mo.md("### Algorithm")
    return


@app.cell
def __(devices, simple_result, torch, zap):
    _i = 0
    _dev = devices[_i]

    _x = simple_result.torchify().power[_i]
    _y_cpu = zap.admm.util.apply_incidence(_dev, _x)
    _y_gpu = zap.admm.util.apply_incidence_gpu(_dev, _x)
    print(torch.linalg.norm(torch.tensor(_y_cpu[0]) - _y_gpu[0]).item())

    _xt = simple_result.torchify().prices
    _y_cpu = zap.admm.util.apply_incidence_transpose(_dev, _xt)
    _y_gpu = zap.admm.util.apply_incidence_transpose_gpu(_dev, _xt)
    print(torch.linalg.norm(torch.tensor(_y_cpu[0]) - _y_gpu[0]).item())
    return


@app.cell(hide_code=True)
def __(simple_devices):
    print("Total Devices: ", sum([d.num_devices for d in simple_devices]))
    return


@app.cell
def __(ADMMSolver, admm_num_iters, eps_pd, rho_angle, rho_power, torch):
    admm = ADMMSolver(
        num_iterations=admm_num_iters,
        rho_power=rho_power,
        rho_angle=rho_angle,
        rtol=eps_pd,
        resid_norm=2,
        safe_mode=False,
        machine="cuda",
        dtype=torch.float32
    )
    print(f"ADMM solving with {admm.machine}")
    return admm,


@app.cell
def __(admm, net, simple_devices, simple_result, time_horizon, torch):
    state, history = admm.solve(
        net,
        simple_devices,
        time_horizon,
        nu_star=torch.tensor(-simple_result.prices, device=admm.machine),
    )
    return history, state


@app.cell
def __(state):
    state.power[0][0].dtype
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("### Results")
    return


@app.cell
def __():
    rho_power = 0.5  # 0.5
    rho_angle = 5.0 * rho_power  # 5.0 * rho_power

    admm_num_iters = 500
    return admm_num_iters, rho_angle, rho_power


@app.cell
def __():
    weighting_strategy = "uniform"
    return weighting_strategy,


@app.cell
def __():
    eps_abs = 1e-3
    return eps_abs,


@app.cell(hide_code=True)
def __(eps_abs, nested_map, np, simple_result, time_horizon):
    _total_num_terminals = sum(
        [sum(x) for x in nested_map(lambda x: x.shape[0], simple_result.power)]
    )
    eps_pd = eps_abs * np.sqrt(_total_num_terminals * time_horizon)
    return eps_pd,


@app.cell(hide_code=True)
def __(admm_num_iters, eps_pd, fstar, np, plt, simple_result):
    def plot_convergence(hist):
        fig, axes = plt.subplots(2, 2, figsize=(7, 4))

        ax = axes[0][0]
        ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("primal residuals")

        ax = axes[0][1]
        ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.dual_power, label="power")
        ax.plot(hist.dual_phase, label="angle")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("dual residuals")

        ax = axes[1][0]
        ax.plot(np.abs(np.array(hist.objective) - fstar) / fstar)
        ax.set_yscale("log")
        ax.set_title("|f - f*| / f*")

        ax = axes[1][1]
        if len(hist.price_error) > 0:
            ax.plot(np.array(hist.price_error) / simple_result.prices.size)
        ax.set_yscale("log")
        ax.set_title("nu - nu*")

        fig.tight_layout()
        return fig
    return plot_convergence,


@app.cell
def __(history, plot_convergence):
    plot_convergence(history)
    return


@app.cell(hide_code=True)
def __():
    # _admm = WeightedADMMSolver(
    #     num_iterations=admm_num_iters,
    #     rho_power=rho_power,
    #     rho_angle=rho_angle,
    #     rtol=eps_pd,
    #     resid_norm=2,
    #     safe_mode=True,
    #     weighting_strategy="uniform",
    # )

    # _state, _history = _admm.solve(
    #     net, simple_devices, time_horizon, nu_star=-simple_result.prices
    # )

    # plot_convergence(_history)

    # admm = WeightedADMMSolver(
    #     num_iterations=admm_num_iters,
    #     rho_power=rho_power,
    #     rho_angle=rho_angle,
    #     rtol=eps_pd,
    #     resid_norm=2,
    #     safe_mode=True,
    #     weighting_strategy=weighting_strategy,
    #     weighting_seed=0,
    # )

    # state, history = admm.solve(
    #     net, simple_devices, time_horizon, nu_star=-simple_result.prices
    # )
    return


@app.cell(hide_code=True)
def __(
    admm,
    fstar,
    history,
    nested_norm,
    rho_power,
    simple_result,
    state,
    torch,
):
    _x = simple_result.torchify(machine=admm.machine)

    print("f/f* =", history.objective[-1] / fstar)
    print(
        "Power Imbalance:",
        torch.linalg.norm(state.avg_power) / nested_norm(_x.power),
    )
    print(
        "Phase Inconsistency:",
        nested_norm(state.resid_phase) / (nested_norm(_x.angle) + 1e-8),
    )
    print(
        "Price Error:",
        torch.linalg.norm(state.dual_power * rho_power + _x.prices, 1)
        / torch.linalg.norm(_x.prices, 1),
    )
    return


@app.cell
def __(admm, plt, rho_power, simple_result, state, torch):
    _x = simple_result.torchify(machine=admm.machine)

    print(torch.linalg.norm(_x.prices + state.dual_power * rho_power, 1))
    print(simple_result.prices.size)
    price_errs = ((_x.prices + state.dual_power * rho_power)).cpu().numpy()

    plt.figure(figsize=(7, 2))
    plt.scatter(range(price_errs.size), price_errs.ravel(), s=1)
    return price_errs,


@app.cell(hide_code=True)
def __(nested_norm, simple_devices, simple_result, torch):
    _x = simple_result.torchify()

    total_power = nested_norm(_x.power)
    total_phase = nested_norm(_x.angle) + 1e-8
    fstar = sum([
        d.operation_cost(_x.power[i], _x.angle[i], None, la=torch)
        for i, d in enumerate(simple_devices)
    ]).item()

    # print(f"Absolute Power Error: {power_errors[-1]}")
    # print(f"Relative Power Error: {power_errors[-1] / total_power}\n")

    # print(f"Absolute Angle Error: {angle_errors[-1]}")
    # print(f"Relative Angle Error: {angle_errors[-1] / total_phase}")
    return fstar, total_phase, total_power


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Debug")
    return


@app.cell
def __():
    # np.sum(np.abs(power_var[3][1])), np.sum(np.abs(phase_var[3][0] - phase_var[3][1]))
    return


@app.cell
def __():
    # _one_power = [np.ones_like(p) for p in power_var[3]]
    # _one_angle = [5 * np.ones_like(_one_power[0]), 3 * np.ones_like(_one_power[0])]

    # p_cvx, a_cvx = simple_devices[3].cvx_admm_prox_update(rho_power, rho_angle, power_var[3], phase_var[3])
    # p_math, a_math = simple_devices[3].admm_prox_update(rho_power, rho_angle, power_var[3], phase_var[3])

    # print(np.linalg.norm(p_math[0] - p_cvx[0]))
    return


@app.cell
def __(np, simple_devices, sp):
    full_incidence = sp.hstack(
        [sp.hstack(d.incidence_matrix) for d in simple_devices]
    ).todense()

    np.sort(np.sum(full_incidence, axis=1).flatten())

    np.linalg.cond(full_incidence @ full_incidence.T, p=2)
    return full_incidence,


@app.cell
def __():
    # np.max(state.phase[3][0] - state.phase[3][1])
    return


@app.cell
def __(np, simple_devices):
    np.max(simple_devices[1].linear_cost)
    return


@app.cell
def __(np):
    np.Inf
    return


@app.cell
def __():
    # # WORKING IMPLEMENTATION BELOW
    # # DO NOT DELETE

    # _T = time_horizon
    # _devices = simple_devices

    # power_var = [d.admm_initialize_power_variables(time_horizon) for d in _devices]
    # # power_var = deepcopy(simple_result.power)

    # phase_var = [d.admm_initialize_angle_variables(time_horizon) for d in _devices]
    # # phase_var = deepcopy(simple_result.angle)

    # num_terminals = get_num_terminals(net, _devices)
    # num_ac_terminals = get_num_terminals(net, _devices, only_ac=True)

    # # power_dual = deepcopy(-simple_result.prices) / rho_power
    # power_dual = dc_average(power_var, net, _devices, _T, num_terminals)

    # # phase_dual = deepcopy(nested_map(lambda x: x / rho_angle, simple_result.phase_duals))
    # phase_dual = [d.admm_initialize_angle_variables(time_horizon) for d in _devices]

    # power_bar = dc_average(power_var, net, _devices, _T, num_terminals)
    # theta_bar = ac_average(phase_var, net, _devices, _T, num_ac_terminals)
    # theta_tilde = get_terminal_residual(phase_var, theta_bar, _devices)
    # power_tilde = get_terminal_residual(power_var, power_bar, _devices)

    # other_history = admm.initialize_history()
    # power_errors = []
    # angle_errors = []

    # for iteration in range(admm_num_iters):
    #     # (1) Proximal updates
    #     for i, dev in enumerate(_devices):
    #         set_power = [
    #             # TODO - Can do this using just terminal indexing (maybe more efficient?)
    #             p - Ai.T @ (power_bar + power_dual)
    #             for p, Ai in zip(power_var[i], dev.incidence_matrix)
    #         ]

    #         if phase_var[i] is None:
    #             set_phase = None
    #         else:
    #             set_phase = [
    #                 Ai.T @ theta_bar - v
    #                 for v, Ai in zip(phase_dual[i], dev.incidence_matrix)
    #             ]

    #         p, a = dev.admm_prox_update(rho_power, rho_angle, set_power, set_phase)
    #         power_var[i] = p
    #         phase_var[i] = a

    #     # Update means / residuals
    #     last_theta_bar = deepcopy(theta_bar)
    #     last_power_tilde = deepcopy(power_tilde)

    #     power_bar = dc_average(power_var, net, _devices, _T, num_terminals)
    #     power_tilde = get_terminal_residual(power_var, power_bar, _devices)

    #     theta_bar = ac_average(phase_var, net, _devices, _T, num_ac_terminals)
    #     theta_tilde = get_terminal_residual(phase_var, theta_bar, _devices)

    #     # (2) Scaled price updates
    #     power_dual = power_dual + power_bar
    #     phase_dual = nested_map(lambda x, y: x + y, phase_dual, theta_tilde)

    #     # Compute errors
    #     power_errors += [get_discrep(power_var, simple_result.power)]
    #     angle_errors += [get_discrep(phase_var, simple_result.angle)]

    #     op_costs = [
    #         d.operation_cost(power_var[i], phase_var[i], None)
    #         for i, d in enumerate(_devices)
    #     ]

    #     other_history.objective += [sum(op_costs)]
    #     other_history.power += [np.linalg.norm(power_bar.ravel(), 2)]
    #     other_history.phase += [nested_norm(theta_tilde, 2)]
    #     other_history.dual_power += [
    #         nested_norm(
    #             nested_map(lambda x, y: rho_power * (x - y), power_tilde, last_power_tilde),
    #             2,
    #         )
    #     ]
    #     other_history.dual_phase += [
    #         np.linalg.norm(rho_angle * (theta_bar - last_theta_bar).ravel(), 2)
    #     ]
    #     other_history.price_error += [
    #         np.linalg.norm(power_dual.ravel() * rho_power + simple_result.prices.ravel(), 2)
    #     ]
    return


if __name__ == "__main__":
    app.run()
