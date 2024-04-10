import marimo

__generated_with = "0.3.10"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import cvxpy as cp

    from copy import deepcopy

    import zap
    from zap import DispatchLayer
    return DispatchLayer, cp, deepcopy, mo, np, zap


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
                [100.0, 100.0, 100.0, 100.0],  # Peaker
                [10.0, 50.0, 50.0, 15.0],  # Solar panel
                [15.0, 15.0, 15.0, 15.0],  # CC Gas
            ]
        ),
        linear_cost=np.array([100.0, 0.5, 40.0]),
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
        linear_cost=0.025 * np.ones(3)
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
def __(DispatchLayer, cp, devices, net, time_horizon):
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
        initial_parameters[name] = getattr(devices[index], attr)

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


app._unparsable_cell(
    r"""
    def inv_objective()

    # inv_objective = 
    """,
    name="__"
)


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


app._unparsable_cell(
    r"""
    problem = zap.planning.PlanningProblem(
        operation_objective=
    )
    """,
    name="__"
)


if __name__ == "__main__":
    app.run()
