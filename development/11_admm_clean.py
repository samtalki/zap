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
    import zap
    return zap,


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn
    seaborn.set_theme()
    return plt, seaborn


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Settings")
    return


@app.cell
def __():
    num_days = 8
    return num_days,


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"## Data")
    return


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


@app.cell
def __(pypsa):
    pn = pypsa.Network()
    pn.import_from_csv_folder("data/pypsa/western/elec_s_500")
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
def __(load_pypsa_network, np, num_days, pn, zap):
    net, devices, time_horizon = load_pypsa_network(pn, time_horizon=24 * num_days)
    _ground = zap.Ground(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        voltage=np.array([0.0]),
    )
    devices += [_ground]
    return devices, net, time_horizon


@app.cell
def __(admm, devices, torch):
    torch.cuda.empty_cache()

    torch_devices = [
        d.torchify(machine=admm.machine, dtype=admm.dtype) for d in devices
    ]
    return torch_devices,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Parameters")
    return


@app.cell
def __(backprop, devices, torch_devices):
    param0 = [{} for d in devices]

    for i in [0, 2, 3]:
        p = torch_devices[i].nominal_capacity.clone().detach()
        if backprop:
            p.requires_grad = True

        param0[i]["nominal_capacity"] = p
    return i, p, param0


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Solve Part 1")
    return


@app.cell
def __():
    from zap.admm import ADMMSolver
    return ADMMSolver,


@app.cell
def __(backprop, torch_devices):
    if backprop:
        print("Enabling gradient tape")
        for d in torch_devices:
            if hasattr(d, "nominal_capacity"):
                d.nominal_capacity.requires_grad = True
    return d,


@app.cell
def __():
    backprop = True
    return backprop,


@app.cell
def __(ADMMSolver, backprop, torch):
    if backprop:
        print("Gradient tape enabled.")

    admm = ADMMSolver(
        num_iterations=10_000,
        rtol=1e-3,
        rho_power=1.0,
        rho_angle=1.5,
        resid_norm=2,
        safe_mode=False,
        machine="cuda",
        dtype=torch.float32,
        battery_window=24,
        battery_inner_iterations=10,
        battery_inner_over_relaxation=1.8,
    )
    return admm,


@app.cell
def __(admm, net, param0, time_horizon, torch_devices):
    state, history = admm.solve(
        net,
        torch_devices,
        time_horizon,
        parameters=param0,
    )
    return history, state


@app.cell
def __(admm, history, np, plot_convergence):
    plot_convergence(history, eps_pd=admm.rtol * np.sqrt(admm.total_terminals))
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


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"## Solve Part 2")
    return


@app.cell
def __(deepcopy, param0, torch):
    param1 = deepcopy(param0)

    for _p in param1:
        for k,v in _p.items():
            _p[k] = v.clone().detach() + torch.rand(v.shape, device="cuda") * 0.010
    return k, param1, v


@app.cell
def __(admm, net, param1, state, time_horizon, torch_devices):
    state1, history1 = admm.solve(
        net,
        torch_devices,
        time_horizon,
        parameters=param1,
        initial_state=state.copy(),
    )
    return history1, state1


@app.cell
def __(admm, history1, np, plot_convergence):
    plot_convergence(history1, eps_pd=admm.rtol * np.sqrt(admm.total_terminals))
    return


if __name__ == "__main__":
    app.run()
