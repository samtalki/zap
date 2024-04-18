import marimo

__generated_with = "0.3.10"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import pandas as pd
    import datetime as dt
    import torch
    import pypsa

    from copy import deepcopy

    import zap
    from zap import DispatchLayer
    return DispatchLayer, cp, deepcopy, dt, mo, np, pd, pypsa, torch, zap


@app.cell
def __(dt, pd):
    pn_dates = pd.date_range(
        dt.datetime(2019, 1, 2, 0),
        dt.datetime(2019, 1, 2, 0) + dt.timedelta(hours=12),
        freq="1h",
        inclusive="left",
    )
    return pn_dates,


@app.cell
def __(np, pn_dates, pypsa, zap):
    mode = "pypsa"

    if mode == "classic":  # Classic settings
        net, devices = zap.importers.load_test_network(
            num_nodes=10, line_type=zap.ACLine
        )
        devices[2].linear_cost *= 0.0

    else:  # PyPSA settings
        pn = pypsa.Network("~/pypsa-usa/workflow/resources/western/elec_s_100.nc")
        net, devices = zap.importers.load_pypsa_network(
            pn, pn_dates, power_unit=1e3, cost_unit=1e3
        )

    devices = devices + [
        zap.Ground(
            num_nodes=net.num_nodes, terminal=np.array([0]), voltage=np.array([7.0])
        )
    ]

    for d in devices:
        print(type(d))
    return d, devices, mode, net, pn


@app.cell
def __(mo):
    mo.md("### Primal Problem")
    return


@app.cell
def __(cp, devices, net):
    y_primal = net.dispatch(devices, add_ground=False, solver=cp.MOSEK)
    print(y_primal.problem.value)
    return y_primal,


@app.cell
def __(mo):
    mo.md("### Dual Problem")
    return


@app.cell
def __(devices, zap):
    dual_devices = zap.dual.dualize(devices)
    return dual_devices,


@app.cell
def __(cp, dual_devices, net):
    y_dual = net.dispatch(dual_devices, dual=True, add_ground=False, solver=cp.MOSEK)
    print(y_dual.problem.value)
    return y_dual,


@app.cell
def __(np, y_dual, y_primal):
    print(y_primal.problem.value + y_dual.problem.value)
    print(np.linalg.norm(y_dual.global_angle - y_primal.prices))
    print(np.linalg.norm(y_primal.global_angle - y_dual.prices))
    return


if __name__ == "__main__":
    app.run()
