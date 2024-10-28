import marimo

__generated_with = "0.3.10"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import pandas as pd

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
        torch,
        zap,
    )


@app.cell
def __():
    import scipy.sparse as sp
    return sp,


@app.cell
def __(mo):
    mo.md("## Network")
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
def __():
    DEFAULT_PYPSA_KWARGS = {
        "marginal_load_value": 1000.0,
        "load_cost_perturbation": 10.0,
        "generator_cost_perturbation": 1.0,
    }
    return DEFAULT_PYPSA_KWARGS,


@app.cell
def __(mo):
    mo.md("## Layer")
    return


@app.cell
def __():
    def dev_index(devices, tp):
        indices = [i for i, dev in enumerate(devices) if isinstance(dev, tp)]
        if len(indices) > 0:
            return indices[0]
        else:
            return None
    return dev_index,


@app.cell
def __(DispatchLayer, cp, dev_index, zap):
    def make_layer(
        net,
        devices,
        time_horizon,
        use_gens=True,
        use_lines=True,
        solver=cp.MOSEK,
        solver_opts={},
    ):
        gen_index = dev_index(devices, zap.Generator)
        line_index = dev_index(devices, zap.ACLine)

        parameter_names = {}
        if use_gens:
            parameter_names["generator_capacity"] = (gen_index, "nominal_capacity")
        if use_lines:
            parameter_names["line_capacity"] = (line_index, "nominal_capacity")

        layer = DispatchLayer(
            net,
            devices,
            parameter_names=parameter_names,
            time_horizon=time_horizon,
            solver=solver,
            solver_kwargs=solver_opts,
        )

        parameter_values = {}
        for name, (index, attr) in parameter_names.items():
            parameter_values[name] = getattr(devices[index], attr)

        return layer, parameter_values
    return make_layer,


@app.cell
def __(np):
    np.random.rand(10) > 0.0
    return


@app.cell
def __(load_pypsa_network, make_layer):
    _F, _theta = make_layer(*load_pypsa_network(time_horizon=4))
    _y = _F(**_theta)
    return


@app.cell
def __(mo):
    mo.md("## Test Layer Gradients")
    return


@app.cell
def __():
    import time
    return time,


@app.cell(hide_code=True)
def __(deepcopy, np, time):
    def test_layer_gradients(layer, theta0, regularize=1e-8, delta=1e-4, param_min=1e-4):
        theta0 = deepcopy(theta0)
        start = time.time()
        y0 = layer(**theta0)
        runtime = time.time() - start
        print("Forward Pass:", runtime)


        # Define linear objective J(y) and it's gradient nabla_J
        dJ = y0.package(np.zeros_like(y0.vectorize()))
        dJ.power[0][0] += layer.devices[0].linear_cost

        def J(y):
            return np.dot(y.vectorize(), dJ.vectorize())

        J0 = J(y0)

        # Compute gradient dJ / dtheta
        start = time.time()
        dJ_dtheta0 = layer.backward(y0, dJ, **theta0, regularize=regularize)
        runtime = time.time() - start
        print("Backward Pass:", runtime)

        # Perturb parameter slightly
        theta1 = deepcopy(theta0)
        for key in theta0.keys():
            theta1[key] += delta * np.ones_like(dJ_dtheta0[key])
            # theta1[key] = np.maximum(theta1[key], param_min)

        # Re-evaluate objective
        y1 = layer(**theta1)
        J1 = J(y1)

        # Compare to predicted change
        J_difference_true = J1 - J0
        J_difference_est = 0.0
        for key in theta0.keys():
            J_difference_est += dJ_dtheta0[key].T @ (theta1[key] - theta0[key])

        abs_error = J_difference_true - J_difference_est
        rel_error = abs_error / np.abs(J_difference_true)

        for key, v in dJ_dtheta0.items():
            print(f"Gradient Norm: {np.linalg.norm(v)}")
            print(f"Step Norm: {np.linalg.norm(theta1[key] - theta0[key])}")

        return {
            "abs_error": abs_error,
            "rel_error": rel_error,
            "J0": J0,
            "J1": J1,
            "delta_J_true": J_difference_true,
            "delta_J_est": J_difference_est,
            "grads": dJ_dtheta0,
        }
    return test_layer_gradients,


@app.cell
def __(load_pypsa_network):
    net, devices, time_horizon = load_pypsa_network(
        num_nodes=100,
        time_horizon=8,
        marginal_load_value=1000.0,
        load_cost_perturbation=100.0,
        generator_cost_perturbation=2.0,
        ac_transmission_cost=1.0,
        exclude_batteries=False,
        cost_unit=1000.0,
        power_unit=1000.0
    )
    return devices, net, time_horizon


@app.cell
def __(cp, devices, make_layer, net, test_layer_gradients, time_horizon):
    _F, _theta = make_layer(
        net,
        devices,
        time_horizon,
        use_lines=True,
        solver=cp.MOSEK,
        solver_opts={"verbose": False, "accept_unknown": True},
    )
    y = _F(**_theta)
    errors = test_layer_gradients(_F, _theta, delta=1e-3, regularize=1e-4)

    errors
    return errors, y


@app.cell
def __(devices, errors, np):
    # DEBUG
    np.max(np.abs(errors["grads"]["generator_capacity"]))

    _x = devices[3].nominal_capacity
    _x = devices[3].susceptance
    np.sort(_x, axis=0)[list(range(5)) + list(range(-6, -1))]
    return


if __name__ == "__main__":
    app.run()
