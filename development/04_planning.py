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
    # net, devices = zap.importers.load_test_network(
    #     num_nodes=10, line_type=line_type
    # )
    net, devices = zap.importers.load_garver_network(line_slack=1.0)

    time_horizon = devices[0].time_horizon
    num_nodes = net.num_nodes
    return devices, net, num_nodes, time_horizon


@app.cell
def __(devices):
    devices[2].slack
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Construct Layer")
    return


@app.cell
def __(DispatchLayer, cp, deepcopy, devices, net, time_horizon):
    parameter_names = {
        # "generator_capacity": (0, "nominal_capacity"),
        "line_capacity": (2, "nominal_capacity"),
        # "battery_capacity": (3, "power_capacity"),
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

    initial_parameters["line_capacity"] += 0.1
    # initial_parameters
    return attr, index, initial_parameters, layer, name, parameter_names


@app.cell
def __(initial_parameters, layer):
    y0 = layer(**initial_parameters)
    return y0,


@app.cell
def __(y0):
    y0.power[0]
    return


@app.cell
def __(initial_parameters, y0):
    list(
        zip(
            initial_parameters["line_capacity"],
            y0.power[2][1],
            y0.local_inequality_duals[2][0],
            y0.local_inequality_duals[2][1]
        )
    )
    return


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
    # other_params["generator_capacity"][1] += 10.0
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
        regularize=1e-6
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
def __(zap):
    line_type = zap.ACLine
    return line_type,


@app.cell
def __(initial_parameters):
    initial_parameters["line_capacity"]
    return


@app.cell
def __(J0, deepcopy, grad, initial_parameters, problem):
    _pname = "line_capacity"
    _delta = 0.001


    for _pind in range(len(initial_parameters[_pname])):

        new_parameters = deepcopy(initial_parameters)
        new_parameters[_pname][_pind] += _delta

        J1 = problem.forward(**new_parameters)

        print(
            _pind,
            initial_parameters[_pname][_pind],
            J1 - J0.detach().numpy() - grad[_pname][_pind].numpy() * _delta,
        )
    return J1, new_parameters


@app.cell
def __(mo):
    mo.md("## Gradient Descent Loop")
    return


@app.cell
def __():
    import cProfile
    import pstats
    return cProfile, pstats


@app.cell
def __():
    # _J0 = problem.forward(**initial_parameters, requires_grad=True)
    # cProfile.run("problem.backward()", sort=pstats.SortKey.CUMULATIVE)
    return


@app.cell
def __():
    # stats = cProfile.run("problem.solve(num_iterations=50)", sort=pstats.SortKey.CUMULATIVE)
    return


@app.cell
def __(initial_parameters, layer):
    layer(**initial_parameters)
    None
    return


@app.cell
def __(problem, zap):
    alg = zap.planning.GradientDescent(step_size=1e-2)

    state, history = problem.solve(num_iterations=50, algorithm=alg)
    return alg, history, state


@app.cell
def __(history):
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_theme(style="whitegrid")

    loss = [h.detach().numpy() for h in history["loss"]]

    print(loss[-1])

    plt.figure(figsize=(6, 2))
    plt.plot(loss)
    plt.show()
    return loss, plt, seaborn


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Debugging")
    return


@app.cell
def __():
    # Learned - backward accumulates gradients!

    # _x = torch.tensor(np.array([1.0]), requires_grad=True)

    # _y1 = _x + _x
    # _y2 = 5 * _x

    # _y1.backward(retain_graph=True)
    # print(_x.grad)

    # _y2.backward(retain_graph=True)
    # print(_x.grad)

    # _z = 3 * _x
    # _z.backward()

    # print(_x.grad

    #      )
    return


if __name__ == "__main__":
    app.run()
