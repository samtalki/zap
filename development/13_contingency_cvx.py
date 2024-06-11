import marimo

__generated_with = "0.6.17"
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


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Data")
    return


@app.cell
def __():
    num_days = 1
    num_nodes = 100
    return num_days, num_nodes


@app.cell(hide_code=True)
def __(mo, num_nodes, pypsa):
    pn = pypsa.Network()
    pn.import_from_csv_folder(f"data/pypsa/western/elec_s_{num_nodes}")
    mo.output.clear()
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


@app.cell(hide_code=True)
def __(load_pypsa_network, np, num_days, pn, zap):
    net, devices, time_horizon = load_pypsa_network(pn, time_horizon=24 * num_days)
    _ground = zap.Ground(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        voltage=np.array([0.0]),
    )
    devices += [_ground]
    return devices, net, time_horizon


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Solve Base Case")
    return


@app.cell(hide_code=True)
def __(cp, devices, net):
    y0 = net.dispatch(devices, solver=cp.MOSEK)
    print("Base Problem Objective Value:", y0.problem.value)
    return y0,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Solve with Contingencies")
    return


@app.cell
def __():
    contingency_device = 3
    num_contingencies = 100
    return contingency_device, num_contingencies


@app.cell(hide_code=True)
def __(contingency_device, devices, num_contingencies, sp):
    contingency_mask = sp.lil_matrix(
        (num_contingencies, devices[contingency_device].num_devices)
    )

    for c in range(num_contingencies):
        contingency_mask[c, c] = 1.0

    contingency_mask = contingency_mask.tocsr()
    return c, contingency_mask


@app.cell(hide_code=True)
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
    print("Contingency Problem Objective Value:", yc.problem.value)
    return yc,


@app.cell
def __(np, yc):
    _c = 4
    np.linalg.norm(yc.power[3][_c + 1][0][_c, :])
    return


if __name__ == "__main__":
    app.run()
