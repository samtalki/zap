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
def __():
    num_nodes = 10
    return num_nodes,


@app.cell
def __(np, num_nodes, zap):
    net = zap.PowerNetwork(num_nodes)

    generators = zap.Generator(
        num_nodes=num_nodes,
        terminal=np.array([0, 1, 3]),
        dynamic_capacity=np.array(
            [
                np.ones(4),  # Peaker
                [0.2, 1.0, 1.0, 0.3],  # Solar panel
                np.ones(4),  # CC Gas
            ]
        ),
        linear_cost=np.array([100.0, 0.5, 40.0]),
        nominal_capacity=np.array([100.0, 50.0, 15.0]),
        capital_cost=np.array([40.0, 50.0, 100.0]),
    )

    loads = zap.Load(
        num_nodes=num_nodes,
        terminal=np.array([0]),
        load=np.array([[30.0, 40.0, 45.0, 80.0]]),
        linear_cost=np.array([200.0]),
    )

    links = zap.ACLine(
        num_nodes=num_nodes,
        source_terminal=np.array([0, 1, 3]),
        sink_terminal=np.array([1, 3, 0]),
        capacity=np.array([45.0, 50.0, 11.0]),
        susceptance=np.array([0.1, 0.05, 1.0]),
        linear_cost=0.025 * np.ones(3),
        capital_cost=np.array([15.0, 25.0, 30.0]),
    )

    batteries = zap.Battery(
        num_nodes=num_nodes,
        terminal=np.array([1]),
        power_capacity=np.array([5.0]),
        duration=np.array([4.0]),
        linear_cost=np.array([0.01]),
    )

    devices = [generators, loads, links, batteries]
    time_horizon = 4
    return batteries, devices, generators, links, loads, net, time_horizon


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
def __(generators, initial_parameters, zap):
    def inv_objective(use_torch=False, **kwargs):
        capital_cost = generators.capital_cost

        if use_torch:
            capital_cost = zap.util.torchify(capital_cost)

        return capital_cost.T @ kwargs["generator_capacity"]

    inv_objective(**initial_parameters)
    return inv_objective,


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
    inv_objective,
    layer,
    lower_bounds,
    op_objective,
    upper_bounds,
    zap,
):
    problem = zap.planning.PlanningProblem(
        operation_objective=op_objective,
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
    _pname = "line_capacity"
    _pind = 1
    _delta = 0.00001

    new_parameters = deepcopy(initial_parameters)
    new_parameters[_pname][_pind] += _delta

    J1 = problem.forward(**new_parameters)

    print(J1 - J0.detach().numpy())
    print(grad[_pname][_pind] * _delta)
    return J1, new_parameters


if __name__ == "__main__":
    app.run()
