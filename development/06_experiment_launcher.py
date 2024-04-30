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
    return relax,


@app.cell
def __(config, problem, relax, runner):
    result = runner.solve_problem(problem, relax, config, **config["optimizer"])
    return result,


@app.cell
def __():
    # _J = problem["problem"]
    # print(_J(**relax["relaxed_parameters"]))
    # print(relax["lower_bound"])
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

    axes[0].plot(result["history"]["rolling_loss"])
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
    mo.md("## Debug")
    return


@app.cell
def __():
    from copy import deepcopy
    return deepcopy,


@app.cell
def __():
    import pandas as pd
    import datetime as dt
    return dt, pd


@app.cell
def __(problem, result):
    _prob = problem["stochastic_problem"]
    _prob.forward(**result["parameters"], batch=[0])
    return


@app.cell
def __():
    import pypsa

    pn = pypsa.Network()
    pn.import_from_csv_folder("./data/pypsa/western/load_medium/elec_s_100_ec/")
    return pn, pypsa


@app.cell
def __(config, pd, pn, runner, zap):
    all_dates = pd.date_range(
        start=runner.PYPSA_START_DAY, periods=runner.TOTAL_PYPSA_HOUR, freq="1h"
    )
    _, year_devices = zap.importers.load_pypsa_network(
        pn, all_dates, **config["data"]["args"]
    )

    every = 24
    renewable_curve = runner.get_total_renewable_curve(
        year_devices, every=every, renewables=["solar", "onwind", "hydro"]
    )
    return all_dates, every, renewable_curve, year_devices


@app.cell
def __(all_dates, every, plt, renewable_curve):
    plt.plot(all_dates[range(0, 8760, every)], renewable_curve)
    return


if __name__ == "__main__":
    app.run()
