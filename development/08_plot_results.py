import marimo

__generated_with = "0.4.7"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import torch
    import importlib

    from pathlib import Path

    import zap
    return Path, importlib, mo, np, torch, zap


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn
    return plt, seaborn


@app.cell
def __(importlib):
    from experiments import runner
    _ = importlib.reload(runner)
    return runner,


@app.cell
def __():
    config_name = "warm_v04"
    return config_name,


@app.cell
def __(config_name, model_version, runner):
    config_list = runner.expand_config(
        runner.load_config(f"experiments/config/{config_name}.yaml")
    )

    config = config_list[model_version]
    return config, config_list


@app.cell
def __(config, runner):
    data = runner.load_dataset(**config["data"])
    devices = data["devices"]
    return data, devices


@app.cell
def __(config, data, runner):
    problem_data = runner.setup_problem(**data, **config["problem"])
    problem = problem_data["stochastic_problem"].subproblems[0]
    return problem, problem_data


@app.cell
def __():
    import json
    return json,


@app.cell
def __(initial_params, json, model_iter, model_version, np):
    if model_iter == 0:
        model_state = initial_params

    else:
        with open(f"./data/results/cost_battery_v01/{model_version:03d}/model_{model_iter:05d}.json", "r") as f:
            model_state = json.load(f)

        _ref_shapes = {k: v.shape for k, v in initial_params.items()}
        model_state = {
            k: np.array(v).reshape(_ref_shapes[k]) for k, v in model_state.items()
        }
    return f, model_state


@app.cell
def __(problem):
    initial_params = problem.initialize_parameters(None)
    return initial_params,


@app.cell
def __(importlib):
    from experiments import plotter
    _ = importlib.reload(plotter)
    return plotter,


@app.cell
def __():
    model_iter = 90
    return model_iter,


@app.cell
def __():
    model_version = 0
    return model_version,


@app.cell
def __(devices, initial_params, model_state, plotter):
    plotter.capacity_plot(initial_params, model_state, devices)[0]
    return


@app.cell
def __(model_state, model_version, plotter, problem, y_model):
    fig, ax = plotter.stackplot(model_state, problem.layer, y_model)
    _bat_cost = 100 + 25 * (model_version - 2)
    ax.set_title(f"Battery Cost: {_bat_cost}%")
    fig.tight_layout()

    fig
    return ax, fig


@app.cell
def __(model_state, problem):
    _prob = problem
    print("System Cost:", _prob(**model_state, batch=[0]))
    print("Investment Cost:", _prob.inv_cost)
    print("Operation Cost:", _prob.op_cost)
    y_model = _prob.state
    return y_model,


if __name__ == "__main__":
    app.run()
