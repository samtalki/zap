import marimo

__generated_with = "0.4.3"
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

    seaborn.set_theme(style="white", rc={
        "font.size" : 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # "axes.ticksize": 8,
    })
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
def __():
    # _d = data["devices"][0]

    # _d.capital_cost[_d.fuel_type == "hydro"]
    return


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
def __(problem, relax):
    if relax is not None:
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
    _fig, _axes = plt.subplots(2, 1, figsize=(8, 3))

    _axes[0].plot(result["history"]["loss"])
    _axes[1].plot(result["history"]["proj_grad_norm"])
    _axes[1].set_yscale("log")

    _fig
    return


@app.cell
def __():
    # runner.save_results(relax, result, config)
    return


@app.cell
def __(mo):
    mo.md("## Plot Results")
    return


@app.cell
def __():
    import json
    return json,


@app.cell
def __(importlib):
    from experiments import plotter
    _ = importlib.reload(plotter)
    return plotter,


@app.cell
def __(json, model_iter, np, problem):
    with open(f"./data/results/base_v03/000/model_{model_iter:05d}.json", "r") as f:
        model_state = json.load(f)

    _ref_shapes = {
        k: v.shape for k, v in problem["problem"].initialize_parameters(None).items()
    }
    model_state = {
        k: np.array(v).reshape(_ref_shapes[k]) for k, v in model_state.items()
    }
    return f, model_state


@app.cell
def __(model_state, problem):
    problem["problem"](**model_state)
    return


@app.cell
def __(data, problem, result):
    _J = problem["problem"]

    p1 = result["parameters"]
    p0 = _J.initialize_parameters(None)
    devices = data["devices"]
    return devices, p0, p1


@app.cell
def __(p0, p1, problem):
    layer = problem["problem"].layer
    y0 = layer(**p0)
    y1 = layer(**p1)
    return layer, y0, y1


@app.cell
def __():
    model_iter = 450
    return model_iter,


@app.cell
def __(devices, p0, p1, plotter):
    plotter.capacity_plot(p0, p1, devices)[0]
    return


@app.cell
def __(layer, p1, plotter, y1):
    plotter.stackplot(p1, layer, y1)
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
def __():
    # gens = devices[0]
    # gen_power = y0.power[0][0]
    # fuels = gens.fuel_type

    # total_gen = np.sum(gen_power[fuels == "hydro", :], axis=0)

    # plt.plot(total_gen)
    return


@app.cell
def __():
    # s, c, d = y1.local_variables[-2]
    # bat = layer.devices[-2]

    # np.max(devices[-2].charge_efficiency)


    # plt.plot(np.sum(s, axis=0))
    # plt.scatter(
    #     np.arange(1, 25),
    #     np.sum(s[:, :-1] + c * bat.charge_efficiency - d, axis=0),
    #     c="red"
    # )
    return


@app.cell
def __():
    # _prob = problem["stochastic_problem"]
    # _prob.forward(**result["parameters"], batch=[0])
    return


@app.cell
def __():
    # import pypsa

    # pn = pypsa.Network()
    # pn.import_from_csv_folder("./data/pypsa/western/load_medium/elec_s_100_ec/")
    return


@app.cell
def __():
    # all_dates = pd.date_range(
    #     start=runner.PYPSA_START_DAY, periods=runner.TOTAL_PYPSA_HOUR, freq="1h"
    # )
    # _, year_devices = zap.importers.load_pypsa_network(
    #     pn, all_dates, **config["data"]["args"]
    # )

    # every = 24
    # renewable_curve = runner.get_total_renewable_curve(
    #     year_devices, every=every, renewables=["solar", "onwind", "hydro"]
    # )
    return


@app.cell
def __():
    # plt.plot(all_dates[range(0, 8736, every)], renewable_curve)
    return


@app.cell
def __():
    # pn.generators[["carrier", "capital_cost"]].groupby("carrier").min().iloc[1:] / 1000
    return


@app.cell
def __():
    # {
    #     f: np.mean(devices[0].dynamic_capacity[devices[0].fuel_type == f, :])
    #     for f in plotter.FUEL_NAMES
    # }
    return


@app.cell
def __():
    # pn.generators_t["p_max_pu"]
    return


if __name__ == "__main__":
    app.run()
