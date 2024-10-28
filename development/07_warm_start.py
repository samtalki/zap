import marimo

__generated_with = "0.4.2"
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

    seaborn.set_theme(style="whitegrid")
    return plt, seaborn


@app.cell
def __(importlib):
    from experiments import runner
    _ = importlib.reload(runner)
    return runner,


@app.cell
def __(runner):
    config = runner.load_config("experiments/config/warm_v01.yaml")
    return config,


@app.cell
def __(plt, result):
    fig, ax = plt.subplots(figsize=(6, 2))

    ax.plot(result["history"]["loss"])

    fig.tight_layout()
    fig
    return ax, fig


@app.cell
def __(config, runner):
    data = runner.load_dataset(config)
    problem = runner.setup_problem(data, config)
    return data, problem


@app.cell
def __(config, problem, runner):
    relax = runner.solve_relaxed_problem(problem, config)
    return relax,


@app.cell
def __(config, problem, relax, runner):
    result = runner.solve_problem(problem, relax, config)
    return result,


@app.cell
def __(problem, relax):
    _J = problem["problem"]
    print("No Upgrades:", _J(**_J.initialize_parameters(None)))
    print("Relaxation:", _J(**relax["relaxed_parameters"]))
    return


@app.cell
def __(problem, result):
    _J = problem["problem"]
    print("Warm Start:", _J(**result["initial_state"]))
    print("Optimized:", _J(**result["parameters"]))
    return


@app.cell
def __(relax):
    print("Lower Bound:", relax["lower_bound"])
    return


@app.cell
def __():
    30_000 / 18_500
    return


@app.cell
def __(np, problem, result):
    _final = result["parameters"]
    _init = problem["problem"].initialize_parameters(None)

    _expansion = {k: np.sum(_final[k] - _init[k]) for k in _final.keys()}

    _expansion
    return


if __name__ == "__main__":
    app.run()
