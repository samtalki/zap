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


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Setup Base Network")
    return


@app.cell
def __(zap):
    num_nodes = 10

    net, devices = zap.importers.load_test_network(num_nodes=num_nodes)

    time_horizon = devices[0].time_horizon
    return devices, net, num_nodes, time_horizon


@app.cell
def __(cp, zap):
    _net, _devices = zap.importers.load_garver_network()
     
    _y = _net.dispatch(
        _devices,
        solver=cp.MOSEK
    )

    _y
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Construct Layer")
    return


@app.cell
def __(DispatchLayer, cp, deepcopy, devices, net, time_horizon):
    parameter_names = {
        "generator_capacity": (0, "nominal_capacity"),
        "line_capacity": (2, "nominal_capacity"),
        "battery_capacity": (3, "power_capacity"),
    }

    layer = DispatchLayer(
        net,
        devices,
        parameter_names=parameter_names,
        time_horizon=time_horizon,
        solver=cp.MOSEK,
        solver_kwargs={"verbose": False, "accept_unknown": True},
    )

    initial_parameters = {}
    for name, (index, attr) in parameter_names.items():
        initial_parameters[name] = deepcopy(getattr(devices[index], attr))

    initial_parameters
    return attr, index, initial_parameters, layer, name, parameter_names


@app.cell
def __(initial_parameters, layer):
    y0 = layer(**initial_parameters)
    return y0,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Construct Planning Problem")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("### Objective")
    return


@app.cell
def __(devices, initial_parameters, layer, net, y0, zap):
    op_objective = zap.planning.DispatchCostObjective(net, devices)

    f0 = op_objective(y0, layer.setup_parameters(**initial_parameters))

    print(f0)
    print(y0.problem.value)
    return f0, op_objective


@app.cell
def __(devices, initial_parameters, layer, y0, zap):
    carbon_objective = zap.planning.EmissionsObjective(devices)

    carbon_objective(y0, layer.setup_parameters(**initial_parameters))
    return carbon_objective,


@app.cell
def __(devices, initial_parameters, zap):
    def simple_inv_objective(use_torch=False, **kwargs):
        capital_cost = devices[0].capital_cost

        if use_torch:
            capital_cost = zap.util.torchify(capital_cost)

        return 0.0  # capital_cost.T @ kwargs["generator_capacity"]

    simple_inv_objective(**initial_parameters)
    return simple_inv_objective,


@app.cell
def __(devices, layer, other_params, zap):
    inv_objective = zap.planning.InvestmentObjective(devices, layer)
    inv_objective(**other_params)
    return inv_objective,


@app.cell
def __(deepcopy, initial_parameters):
    other_params = deepcopy(initial_parameters)
    other_params["generator_capacity"][1] += 10.0
    return other_params,


@app.cell(hide_code=True)
def __(mo):
    mo.md("### Problem")
    return


@app.cell
def __():
    max_expansion = 10.0
    return max_expansion,


@app.cell
def __(deepcopy, initial_parameters, max_expansion):
    lower_bounds = deepcopy(initial_parameters)
    upper_bounds = deepcopy(lower_bounds)

    for param, v in upper_bounds.items():
        upper_bounds[param] = v * max_expansion
    return lower_bounds, param, upper_bounds, v


@app.cell
def __(
    carbon_objective,
    inv_objective,
    layer,
    lower_bounds,
    upper_bounds,
    zap,
):
    problem = zap.planning.PlanningProblem(
        operation_objective=carbon_objective,
        investment_objective=inv_objective,
        layer=layer,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        regularize=1e-4
    )
    return problem,


@app.cell
def __(initial_parameters, problem):
    problem(**initial_parameters)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Compute Gradients")
    return


@app.cell
def __(initial_parameters, problem):
    # J0 = problem.forward(**initial_parameters, requires_grad=True)
    # grad = problem.backward()

    J0, grad = problem.forward_and_back(**initial_parameters)

    grad
    return J0, grad


@app.cell(hide_code=True)
def __(mo):
    mo.md("### Gradient Update")
    return


@app.cell
def __(J0, deepcopy, grad, initial_parameters, problem):
    _pname = "generator_capacity"
    _pind = 1
    _delta = 0.001

    new_parameters = deepcopy(initial_parameters)
    new_parameters[_pname][_pind] += _delta

    J1 = problem.forward(**new_parameters)

    print(J1 - J0.detach().numpy())
    print(grad[_pname][_pind] * _delta)
    return J1, new_parameters


if __name__ == "__main__":
    app.run()
