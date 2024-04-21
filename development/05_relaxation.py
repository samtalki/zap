import marimo

__generated_with = "0.4.2"
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
    # pn = pypsa.Network("~/pypsa-usa/workflow/resources/western/elec_s_100.nc")
    pn_dates = pd.date_range(
        dt.datetime(2019, 1, 2, 0),
        dt.datetime(2019, 1, 2, 0) + dt.timedelta(hours=6),
        freq="1h",
        inclusive="left",
    )
    return pn_dates,


@app.cell
def __(mo):
    mo.md("## Dual Dispatch")
    return


@app.cell
def __(np, pn_dates, zap):
    mode = "classic"
    classic_line_type = zap.ACLine

    if mode == "classic":  # Classic settings
        net, devices = zap.importers.load_test_network(
            num_nodes=10, line_type=classic_line_type
        )
        devices[2].linear_cost *= 0.0

    else:  # PyPSA settings
        net, devices = zap.importers.load_pypsa_network(
            None, pn_dates, power_unit=1e3, cost_unit=10.0
        )

    devices = devices + [
        zap.Ground(
            num_nodes=net.num_nodes, terminal=np.array([0]), voltage=np.array([7.0])
        )
    ]

    for d in devices:
        print(type(d))
    return classic_line_type, d, devices, mode, net


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


@app.cell
def __(np, y_primal):
    np.linalg.norm(y_primal.prices)
    return


@app.cell
def __(mo):
    mo.md("## Relaxation")
    return


@app.cell
def __(DispatchLayer, classic_line_type, cp, deepcopy, devices, net, zap):
    _gind = next(i for i, d in enumerate(devices) if isinstance(d, zap.Generator))
    _lind = next(i for i, d in enumerate(devices) if isinstance(d, classic_line_type))
    _bind = next(i for i, d in enumerate(devices) if isinstance(d, zap.Battery))

    parameter_names = {
        "generator_capacity": (_gind, "nominal_capacity"),
        "line_capacity": (_lind, "nominal_capacity"),
        "battery_capacity": (_bind, "power_capacity"),
    }

    initial_parameters = {}
    for name, (index, attr) in parameter_names.items():
        initial_parameters[name] = deepcopy(getattr(devices[index], attr))

    layer = DispatchLayer(
        net,
        devices,
        parameter_names=parameter_names,
        time_horizon=devices[0].time_horizon,
        solver=cp.MOSEK,
        solver_kwargs={"verbose": False, "accept_unknown": True},
        add_ground=False,
    )

    # initial_parameters
    return attr, index, initial_parameters, layer, name, parameter_names


@app.cell
def __(devices, layer, net, zap):
    op_objective = zap.planning.DispatchCostObjective(net, devices)
    inv_objective = zap.planning.InvestmentObjective(devices, layer)
    return inv_objective, op_objective


@app.cell
def __(inv_objective, layer, op_objective, zap):
    problem = zap.planning.PlanningProblem(
        operation_objective=op_objective,
        investment_objective=inv_objective,
        layer=layer,
        lower_bounds=None,
        upper_bounds=None,
        regularize=1e-6,
    )
    return problem,


@app.cell
def __(cp, problem):
    net_var = {p: cp.Variable(lower.shape) for p, lower in problem.lower_bounds.items()}

    lower_bounds = [
        net_var[p] <= problem.upper_bounds[p]
        for p in sorted(net_var.keys())
    ]

    print(sorted(net_var.keys()))
    return lower_bounds, net_var


@app.cell
def __(problem, zap):
    relaxation = zap.planning.RelaxedPlanningProblem(
        problem,
        # solver=cp.MOSEK,
        # solver_kwargs={"verbose": False, "accept_unknown": True},
    )
    relaxed_parameters, data = relaxation.solve()
    return data, relaxation, relaxed_parameters


@app.cell
def __(data):
    print(data["operation_objective"].value, data["investment_objective"].value)
    print(data["operation_objective"].value + data["investment_objective"].value)
    print(data["problem"].value)
    return


@app.cell
def __(initial_parameters, problem, relaxed_parameters):
    print(problem(**initial_parameters))
    print(problem(**relaxed_parameters))
    return


@app.cell
def __(initial_parameters, relaxed_parameters):
    initial_parameters, relaxed_parameters
    return


if __name__ == "__main__":
    app.run()
