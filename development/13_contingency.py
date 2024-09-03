import marimo

__generated_with = "0.8.3"
app = marimo.App(app_title="SCOPF - CVX")


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


@app.cell
def __(devices):
    devices[3].num_devices
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Data""")
    return


@app.cell
def __():
    num_days = 1
    num_nodes = 100
    return num_days, num_nodes


@app.cell
def __(mo, num_nodes, pypsa):
    pn = pypsa.Network()
    pn.import_from_csv_folder(f"data/pypsa/western/elec_s_{num_nodes}")
    mo.output.clear()
    return pn,


@app.cell
def __():
    DEFAULT_PYPSA_KWARGS = {
        "marginal_load_value": 500.0,
        "load_cost_perturbation": 50.0,
        "generator_cost_perturbation": 1.0,
        "cost_unit": 100.0,  # 1000.0,
        "power_unit": 1000.0,
        "b_factor": 1.0,
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
def __(devices):
    torch_devices = [d.torchify(machine="cuda") for d in devices]
    return torch_devices,


@app.cell(hide_code=True)
def __(np, plt):
    def plot_convergence(hist, admm, fstar):
        fig, axes = plt.subplots(2, 2, figsize=(7, 4))

        admm_num_iters = len(hist.objective)
        eps_pd = admm.primal_tol

        ax = axes[0][0]
        total_primal = np.sqrt(np.power(hist.power, 2) + np.power(hist.phase, 2))
        ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.plot(total_primal, color="black", ls="dashed")
        ax.set_yscale("log")
        ax.set_title("primal residuals")

        ax = axes[0][1]
        total_dual = np.sqrt(np.power(hist.dual_power, 2) + np.power(hist.dual_phase, 2))
        ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.dual_power, label="power")
        ax.plot(hist.dual_phase, label="angle")
        ax.plot(total_dual, color="black", ls="dashed")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("dual residuals")

        ax = axes[1][0]
        ax.plot(np.abs(np.array(hist.objective) - fstar) / fstar)
        ax.set_yscale("log")
        ax.set_title("|f - f*| / f*")

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
    mo.md("""## Solve Base Case""")
    return


@app.cell
def __():
    from zap.admm import ADMMSolver
    return ADMMSolver,


@app.cell
def __(ADMMSolver, torch):
    admm = ADMMSolver(
        num_iterations=1000,
        rho_power=1.0,
        rho_angle=1.0,
        adaptive_rho=True,
        rtol=1e-3,
        resid_norm=2,
        machine="cuda",
        dtype=torch.float32,
        battery_window=24,
    )
    return admm,


@app.cell
def __(cp, devices, net):
    y0 = net.dispatch(devices, solver=cp.MOSEK)
    return y0,


@app.cell
def __(admm, net, time_horizon, torch_devices):
    s0, history0 = admm.solve(net, torch_devices, time_horizon)
    return history0, s0


@app.cell(hide_code=True)
def __(history0, np, y0):
    print(
        np.abs(y0.problem.value - history0.objective[-1]) / y0.problem.value * 100,
        "%",
    )
    return


@app.cell(hide_code=True)
def __(history0, y0):
    print("Objective Value (CVX):\t", y0.problem.value)
    print("Objective Value (ADMM):\t", history0.objective[-1])
    return


@app.cell(hide_code=True)
def __(s0, time_horizon, torch):
    _tot_power = torch.sum(torch.abs(s0.power[1][0])) / time_horizon
    _x = torch.mean(torch.sum(torch.abs(s0.avg_power), dim=0)) * 1000.0

    print("Average Power Imbalance per Timestep:\t", _x.item(), "MW")
    print("Average Load per Timestep:\t\t\t\t", _tot_power.item() * 1000.0, "MW")
    print(
        "Relative Imbalance:\t\t\t\t\t\t",
        _x.item() / _tot_power.item() / 1000.0 * 100.0,
        "%",
    )
    return


@app.cell(hide_code=True)
def __(admm, history0, plot_convergence, y0):
    plot_convergence(history0, admm, y0.problem.value)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Solve with Contingencies""")
    return


@app.cell
def __(devices):
    contingency_device = 3
    num_contingencies = 10

    print(type(devices[contingency_device]))
    return contingency_device, num_contingencies


@app.cell
def __(contingency_device, devices, num_contingencies, sp):
    contingency_mask = sp.lil_matrix(
        (num_contingencies, devices[contingency_device].num_devices)
    )

    for c in range(num_contingencies):
        contingency_mask[c, c] = 1.0

    contingency_mask = contingency_mask.tocsr()
    return c, contingency_mask


@app.cell
def __(
    contingency_device,
    contingency_mask,
    cp,
    devices,
    net,
    num_contingencies,
):
    yc = net.dispatch(
        devices,
        solver=cp.MOSEK,
        num_contingencies=num_contingencies,
        contingency_device=contingency_device,
        contingency_mask=contingency_mask,
    )
    return yc,


@app.cell
def __():
    # _c = 0
    # yc.power[3][_c + 1][0][_c, :]
    return


@app.cell
def __(ADMMSolver, torch):
    admm_c = ADMMSolver(
        num_iterations=20_000,
        rho_power=1.5,
        rho_angle=1.5,
        # adaptive_rho=True,
        rtol=1.0e-4,
        resid_norm=2,
        machine="cuda",
        dtype=torch.float32,
        battery_window=24,
        minimum_iterations=10,
    )
    return admm_c,


@app.cell
def __(contingency_mask, torch):
    torch_mask = torch.tensor(contingency_mask.todense(), device="cuda", dtype=torch.float32)
    torch_mask = torch.vstack(
        [
            torch.zeros(torch_mask.shape[1], device="cuda", dtype=torch.float32),
            torch_mask,
        ]
    )
    return torch_mask,


@app.cell
def __(
    admm_c,
    contingency_device,
    net,
    num_contingencies,
    time_horizon,
    torch_devices,
    torch_mask,
):
    sc, historyc = admm_c.solve(
        net,
        torch_devices,
        time_horizon,
        num_contingencies=num_contingencies,
        contingency_device=contingency_device,
        contingency_mask=torch_mask,
    )
    return historyc, sc


@app.cell
def __(historyc, yc):
    print("Objective Value (CVX):\t", yc.problem.value)
    print("Objective Value (ADMM):\t", historyc.objective[-1])
    return


@app.cell
def __(admm_c, historyc, plot_convergence, yc):
    plot_convergence(historyc, admm_c, yc.problem.value)
    return


@app.cell(hide_code=True)
def __(devices):
    for d in devices:
        print(type(d), d.num_devices)
    print(sum(d.num_devices for d in devices))
    return d,


if __name__ == "__main__":
    app.run()
