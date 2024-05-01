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
def __(runner):
    config_list = runner.expand_config(
        runner.load_config("experiments/config/test_default.yaml")
    )

    config = config_list[0]
    return config, config_list


@app.cell
def __(config, runner):
    data = runner.load_dataset(**config["data"])

    devices = data["devices"]
    return data, devices


@app.cell
def __(config, data, runner):
    problem_data = runner.setup_problem(**data, **config["problem"])

    problem = problem_data["problem"]
    return problem, problem_data


@app.cell
def __():
    import json
    return json,


@app.cell
def __(initial_params, json, model_iter, np):
    if model_iter == 0:
        model_state = initial_params

    else:
        with open(f"./data/results/base_v05/000/model_{model_iter:05d}.json", "r") as f:
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
    model_iter = 0
    return model_iter,


@app.cell
def __(devices, initial_params, model_state, plotter):
    plotter.capacity_plot(initial_params, model_state, devices)[0]
    return


@app.cell(hide_code=True)
def __(model_state, problem):
    print("System Cost:", problem(**model_state))
    print("Investment Cost:", problem.inv_cost)
    print("Operation Cost:", problem.op_cost)
    y_model = problem.state
    return y_model,


@app.cell
def __(model_state, plotter, problem, y_model):
    plotter.stackplot(model_state, problem.layer, y_model)
    return


if __name__ == "__main__":
    app.run()
