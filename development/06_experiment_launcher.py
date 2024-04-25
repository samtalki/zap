import marimo

__generated_with = "0.4.2"
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
    config = runner.load_config("experiments/config/default.yaml")
    return config,


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
    return relax,


@app.cell
def __(problem, relax, result):
    _J = problem["problem"]

    print(_J(**_J.initialize_parameters(None)))
    print(_J(**relax["relaxed_parameters"]))
    print(_J(**result["parameters"]))

    print(relax["lower_bound"])
    return


@app.cell
def __(config, problem, relax, runner):
    result = runner.solve_problem(problem, relax, config, **config["optimizer"])
    return result,


@app.cell
def __(plt, relax, result):
    fig, axes = plt.subplots(2, 1, figsize=(8, 3))

    axes[0].plot(result["history"]["loss"] / relax["lower_bound"])
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
def __(deepcopy, problem):
    _prob1 = deepcopy(problem["problem"])
    _prob2 = deepcopy(_prob1)

    _stoch_prob = 0.4 * _prob1 + 0.6 * _prob2

    # Check forward pass
    # _stoch_prob(**result["parameters"])

    # Check full forward and back
    # _J, _grad = _stoch_prob.forward_and_back(**result["parameters"])

    # Check solve
    # np.all(_stoch_prob.lower_bounds["generator"] == _prob1.lower_bounds["generator"])
    # p, hist = _stoch_prob.solve(num_iterations=3)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
