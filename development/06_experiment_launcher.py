import marimo

__generated_with = "0.4.3"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import importlib

    from pathlib import Path

    import zap
    return Path, importlib, mo, np, plt, torch, zap


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
    return data,


@app.cell
def __(config, data, runner):
    problem = runner.setup_problem(**data, **config["problem"])
    return problem,


@app.cell
def __(config, problem, runner):
    relax = runner.solve_relaxed_problem(problem, **config["relaxation"])
    # relax = None
    return relax,


@app.cell
def __(config, problem, relax, runner):
    result = runner.solve_problem(problem, relax, config, **config["optimizer"])
    return result,


@app.cell
def __(problem, relax):
    _J = problem["problem"]

    print(_J(**relax["relaxed_parameters"]))
    print(relax["lower_bound"])
    return


@app.cell
def __(problem, result):
    _J = problem["problem"]

    print(_J(**_J.initialize_parameters(None)))
    print(_J(**result["parameters"]))
    return


@app.cell
def __(plt, result):
    fig, axes = plt.subplots(2, 1, figsize=(8, 3))

    axes[0].plot(result["history"]["loss"])
    axes[1].plot(result["history"]["proj_grad_norm"])
    axes[1].set_yscale("log")

    fig
    return axes, fig


@app.cell
def __():
    # runner.save_results(relax, result, config)
    return


@app.cell
def __(mo):
    mo.md("## Stochastic Problem")
    return


@app.cell
def __():
    from copy import deepcopy
    return deepcopy,


@app.cell
def __(problem, result):
    _prob = problem["stochastic_problem"]

    _prob.forward(**result["parameters"], batch=[0])
    return


@app.cell
def __(zap):
    zap.planning.problem.get_next_batch([2, 3], 2, 4)
    return


if __name__ == "__main__":
    app.run()
