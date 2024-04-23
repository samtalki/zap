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
def __(config, runner):
    data = runner.load_dataset(config)
    return data,


@app.cell
def __(importlib):
    from experiments import runner
    _ = importlib.reload(runner)
    return runner,


@app.cell
def __(config):
    config["name"]
    return


@app.cell
def __(runner):
    config = runner.load_config("experiments/config/default.yaml")
    return config,


@app.cell
def __(config, data, runner):
    problem = runner.setup_problem(data, config)
    return problem,


@app.cell
def __(config, problem, runner):
    relax = runner.solve_relaxed_problem(problem, config)
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
    result = runner.solve_problem(problem, relax, config)
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
    # result["history"]
    return


@app.cell
def __():
    # runner.save_results(relax, result, config)
    return


@app.cell
def __():
    # import pypsa
    return


@app.cell
def __():
    # _csv_dir = f"elec_s_{100}"
    # _csv_dir += "_ec"

    # pn = pypsa.Network()
    # pn.import_from_csv_folder(runner.DATA_PATH / "pypsa/western/" / _csv_dir)
    return


@app.cell
def __(np):
    type(np.argmax([1, 10, 3, 10]).item())
    return


if __name__ == "__main__":
    app.run()
