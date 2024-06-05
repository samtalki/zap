import marimo

__generated_with = "0.6.13"
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
    seaborn.set_theme()
    return plt, seaborn


@app.cell
def __():
    import zap
    return zap,


@app.cell
def __():
    num_days = 2
    return num_days,


@app.cell
def __(torch):
    MACHINE = "cuda"
    DTYPE = torch.float32
    return DTYPE, MACHINE


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Data")
    return


@app.cell(hide_code=True)
def __(pypsa):
    DEFAULT_PYPSA_KWARGS = {
        "marginal_load_value": 500.0,
        "load_cost_perturbation": 10.0,
        "generator_cost_perturbation": 1.0,
        "cost_unit": 1.0,
        "power_unit": 1.0e3,
        "drop_empty_generators": False,
        "expand_empty_generators": 0.5,
    }
    PN = pypsa.Network()
    PN.import_from_csv_folder("data/pypsa/western/load_medium/elec_s_100_ec")
    return DEFAULT_PYPSA_KWARGS, PN


@app.cell(hide_code=True)
def __(DEFAULT_PYPSA_KWARGS, PN, deepcopy, dt, pd, zap):
    def load_pypsa_network(
        pn,
        time_horizon=1,
        start_date=dt.datetime(2019, 1, 2, 0),
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

        net, devices = zap.importers.load_pypsa_network(PN, dates, **all_kwargs)

        return net, devices, time_horizon
    return load_pypsa_network,


@app.cell(hide_code=True)
def __(PN, load_pypsa_network, np, num_days, zap):
    net, devices, time_horizon = load_pypsa_network(PN, time_horizon=24 * num_days)
    _ground = zap.Ground(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        voltage=np.array([0.0]),
    )
    devices += [_ground]
    return devices, net, time_horizon


@app.cell(hide_code=True)
def __(devices, time_horizon):
    print(
        f"{sum(d.num_devices for d in devices)} total devices and {time_horizon} time periods.\n"
    )

    for d in devices:
        print(f"{type(d)} has {d.num_devices} devices")
    return d,


@app.cell(hide_code=True)
def __(DTYPE, MACHINE, devices, torch):
    torch.cuda.empty_cache()
    torch_devices = [d.torchify(machine=MACHINE, dtype=DTYPE) for d in devices]
    return torch_devices,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Layer")
    return


@app.cell(hide_code=True)
def __(np, plt):
    def plot_convergence(hist, eps_pd=None):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2))

        print(f"Primal Resid: {hist.power[-1] + hist.phase[-1]}")
        print(f"Dual Resid: {hist.dual_power[-1] + hist.dual_phase[-1]}")
        print(f"Objective: {hist.objective[-1]}")

        admm_num_iters = len(hist.power)

        ax = axes[0]
        if eps_pd is not None:
            ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title("primal residuals")

        ax = axes[1]
        if eps_pd is not None:
            ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.dual_power, label="power")
        ax.plot(hist.dual_phase, label="angle")
        ax.set_yscale("log")
        # ax.legend(fontsize=8)
        ax.set_title("dual residuals")

        ax = axes[2]
        ax.plot(np.array(hist.objective))
        # ax.set_yscale("log")
        ax.set_title("f")

        # ax = axes[1][1]
        # if len(hist.price_error) > 0:
        #     ax.plot(np.array(hist.price_error) / simple_result.prices.size)
        # ax.set_yscale("log")
        # ax.set_title("nu - nu*")

        fig.tight_layout()
        return fig
    return plot_convergence,


@app.cell
def __():
    def torch_copy(param):
        return {k: v.detach().clone() for k, v in param.items()}
    return torch_copy,


@app.cell
def __():
    from zap.admm import ADMMLayer, ADMMSolver
    return ADMMLayer, ADMMSolver


@app.cell
def __():
    parameter_names = {
        "generator_capacity": (0, "nominal_capacity"),
        "dc_line_capacity": (2, "nominal_capacity"),
        "ac_line_capacity": (3, "nominal_capacity"),
    }
    return parameter_names,


@app.cell
def __(ADMMSolver, DTYPE, MACHINE):
    admm = ADMMSolver(
        num_iterations=2000,
        rtol=1e-2,
        rho_power=1.0,
        rho_angle=1.0,
        resid_norm=2,
        machine=MACHINE,
        dtype=DTYPE,
        battery_window=24,
        battery_inner_iterations=10,
        battery_inner_over_relaxation=1.8,
    )
    return admm,


@app.cell
def __(ADMMLayer, admm, net, parameter_names, time_horizon, torch_devices):
    layer = ADMMLayer(net, torch_devices, parameter_names, time_horizon, admm)
    return layer,


@app.cell
def __(layer):
    theta0 = layer.initialize_parameters(requires_grad=True)
    return theta0,


@app.cell
def __(admm, layer, np, plot_convergence, theta0, torch_copy):
    y0 = layer(**torch_copy(theta0))
    plot_convergence(layer.history, eps_pd=admm.rtol * np.sqrt(admm.total_terminals))
    return y0,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Planning Problem")
    return


@app.cell
def __():
    from zap.planning import PlanningProblem
    return PlanningProblem,


@app.cell
def __(PlanningProblem, layer, net, torch_devices, zap):
    op_cost = zap.planning.DispatchCostObjective(net, torch_devices)
    inv_cost = zap.planning.InvestmentObjective(torch_devices, layer)

    problem = PlanningProblem(op_cost, inv_cost, layer)
    return inv_cost, op_cost, problem


@app.cell
def __(MACHINE, theta0, torch, torch_copy):
    theta1 = torch_copy(theta0)
    for k,v in theta1.items():
        theta1[k] = v + 0.05 * torch.rand(v.shape, device=MACHINE)
    return k, theta1, v


@app.cell
def __(problem, theta1, torch_copy, y0):
    # c0 = problem(**torch_copy(theta0), initial_state=y0.copy())
    c1 = problem(**torch_copy(theta1), requires_grad=True, initial_state=y0.copy())
    grad1 = problem.backward()
    return c1, grad1


@app.cell
def __(grad1):
    grad1["generator_capacity"][:10]
    return


@app.cell
def __(mo):
    mo.md("## Solve with Gradient Descent")
    return


if __name__ == "__main__":
    app.run()
