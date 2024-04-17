import marimo

__generated_with = "0.3.10"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import torch

    from copy import deepcopy

    import zap
    from zap import DispatchLayer
    return DispatchLayer, cp, deepcopy, mo, np, torch, zap


@app.cell
def __(np, zap):
    net, devices = zap.importers.load_test_network(num_nodes=10, line_type=zap.DCLine)

    devices = devices[:2]
    devices += [zap.Ground(
        num_nodes=net.num_nodes, terminal=np.array([0]), voltage=np.array([7.0])
    )]

    print([type(d) for d in devices])
    return devices, net


@app.cell
def __(mo):
    mo.md("### Primal Problem")
    return


@app.cell
def __(devices, net):
    y_primal = net.dispatch(devices, add_ground=False)
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
def __(dual_devices, net):
    y_dual = net.dispatch(dual_devices, dual=True, add_ground=False)
    print(y_dual.problem.value)
    return y_dual,


@app.cell
def __(y_dual, y_primal):
    y_dual.global_angle, y_primal.prices
    return


@app.cell
def __(y_dual, y_primal):
    y_primal.global_angle, -y_dual.prices
    return


if __name__ == "__main__":
    app.run()
