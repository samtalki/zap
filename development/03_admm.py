import marimo

__generated_with = "0.3.4"
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

    import zap
    from zap import DispatchLayer
    return (
        DispatchLayer,
        cp,
        deepcopy,
        dt,
        importlib,
        mo,
        np,
        pd,
        pypsa,
        sp,
        torch,
        zap,
    )


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
    pn = pypsa.Network(
        f"~/pypsa-usa/workflow/resources/western/elec_s_100_ec_lv1.25_Co2L1.25.nc"
    )
    return pn,


@app.cell
def __(np, pn):
    np.max((pn.generators["p_nom"] - pn.generators["p_nom_min"]).values)
    # [["p_nom", "p_nom_min",  "p_nom_max", "p_nom_extendable"]]

    pn.generators
    return


@app.cell
def __(DEFAULT_PYPSA_KWARGS, deepcopy, dt, pd, pypsa, zap):
    def load_pypsa_network(
        time_horizon=1,
        num_nodes=100,
        start_date=dt.datetime(2019, 1, 2, 0),
        exclude_batteries=False,
        **pypsa_kwargs,
    ):
        all_kwargs = deepcopy(DEFAULT_PYPSA_KWARGS)
        all_kwargs.update(pypsa_kwargs)
        print(all_kwargs)

        pn = pypsa.Network(
            f"~/pypsa-usa/workflow/resources/western/elec_s_{num_nodes}.nc"
        )
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
def __(load_pypsa_network):
    net, devices, time_horizon = load_pypsa_network(time_horizon=8)

    for _d in devices:
        print(type(_d))
    return devices, net, time_horizon


@app.cell
def __(cp, devices, net, time_horizon):
    result = net.dispatch(
        devices,
        time_horizon,
        solver=cp.MOSEK,
        add_ground=False,
        solver_kwargs={"verbose": False}
    )
    result.problem.value
    return result,


@app.cell
def __(
    devices,
    get_nodal_average,
    get_terminal_residual,
    nested_norm,
    net,
    np,
    result,
    time_horizon,
):
    # Compute global power / phase imbalance
    average_power = get_nodal_average(result.power, net, devices, time_horizon)
    average_angle = get_nodal_average(
        result.angle, net, devices, time_horizon, only_ac=True
    )
    global_phase_imbalance = average_angle - result.global_angle

    print(f"Power Imbalance: {np.linalg.norm(average_power, 1)}")
    print(f"Global Phase Imbalance: {np.linalg.norm(global_phase_imbalance, 1)}")

    # Compute local phase imbalance
    phase_residual = get_terminal_residual(result.angle, average_angle, devices)

    print(f"Local Phase Imbalance: {nested_norm(phase_residual)}")
    return (
        average_angle,
        average_power,
        global_phase_imbalance,
        phase_residual,
    )


@app.cell
def __(mo):
    mo.md("## Solve with CVXPY")
    return


@app.cell
def __(cp, deepcopy, devices, nested_norm, net, time_horizon, zap):
    simple_devices = deepcopy(devices[:3])
    use_ac = False

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

    # # Add ground
    # _ground = zap.Ground(
    #     num_nodes=net.num_nodes, terminal=np.array([0]), voltage=np.array([0.0])
    # )
    # simple_devices += [_ground]

    # Dispatch
    simple_result = net.dispatch(
        simple_devices,
        time_horizon,
        solver=cp.MOSEK,
        add_ground=False,
    )

    print(nested_norm(simple_result.angle))
    print(nested_norm(simple_result.power))
    for _d in simple_devices:
        print(type(_d))
    return simple_devices, simple_result, use_ac


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
def __(
    WeightedADMMSolver,
    admm_num_iters,
    eps_pd,
    net,
    rho_angle,
    rho_power,
    simple_devices,
    simple_result,
    time_horizon,
    weighting_strategy,
):
    admm = WeightedADMMSolver(
        num_iterations=admm_num_iters,
        rho_power=rho_power,
        rho_angle=rho_angle,
        rtol=eps_pd,
        resid_norm=2,
        safe_mode=True,
        weighting_strategy=weighting_strategy,
        weighting_seed=0,
    )

    state, history = admm.solve(
        net, simple_devices, time_horizon, nu_star=-simple_result.prices
    )
    return admm, history, state


@app.cell
def __(mo):
    mo.md("### Results")
    return


@app.cell
def __():
    weighting_strategy = "random"

    rho_power = 0.5
    rho_angle = 5.0 * rho_power

    admm_num_iters = 300
    return admm_num_iters, rho_angle, rho_power, weighting_strategy


@app.cell
def __(nested_map, np, simple_result, time_horizon):
    eps_abs = 1e-3
    _total_num_terminals = sum(
        [sum(x) for x in nested_map(lambda x: x.shape[0], simple_result.power)]
    )
    eps_pd = eps_abs * np.sqrt(_total_num_terminals * time_horizon)
    eps_pd
    return eps_abs, eps_pd


@app.cell(hide_code=True)
def __(admm_num_iters, eps_pd, fstar, np, plt, state):
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
        ax.plot(np.abs(hist.objective - fstar) / fstar)
        ax.set_yscale("log")
        ax.set_title("|f - f*| / f*")

        ax = axes[1][1]
        if len(hist.price_error) > 0:
            ax.plot(np.array(hist.price_error) / state.dual_power.size)
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
def __(fstar, history, nested_norm, np, rho_power, simple_result, state):
    print("f/f* =", history.objective[-1] / fstar)
    print(
        "Power Imbalance:", np.linalg.norm(state.avg_power, 1) / nested_norm(simple_result.power)
    )
    print(
        "Phase Inconsistency:",
        nested_norm(state.resid_phase) / (nested_norm(simple_result.angle) + 1e-8),
    )
    print(
        "Price Error:",
        np.linalg.norm(state.dual_power * rho_power + simple_result.prices, 1)
        / np.linalg.norm(simple_result.prices, 1),
    )
    return


@app.cell(hide_code=True)
def __(np, plt, rho_power, simple_result, state):
    print(
        np.linalg.norm(simple_result.prices + state.dual_power * rho_power, 1)
        / simple_result.prices.size
    )
    price_errs = (simple_result.prices + state.dual_power * rho_power) / np.maximum(
        np.abs(simple_result.prices), 1.0
    )
    plt.scatter(range(price_errs.size), price_errs.flatten(), s=4)
    return price_errs,


@app.cell
def __(nested_norm, simple_devices, simple_result):
    total_power = nested_norm(simple_result.power)
    total_phase = nested_norm(simple_result.angle) + 1e-8
    fstar = sum([
        d.operation_cost(simple_result.power[i], simple_result.angle[i], None)
        for i, d in enumerate(simple_devices)
    ])

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
def __(np, state):
    np.max(state.phase[3][0] - state.phase[3][1])
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
