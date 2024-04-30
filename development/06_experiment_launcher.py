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
def __(importlib):
    from experiments import plotter
    _ = importlib.reload(plotter)
    return plotter,


@app.cell
def __(data, problem, result):
    _J = problem["problem"]

    p1 = result["parameters"]
    p0 = _J.initialize_parameters(None)
    devices = data["devices"]
    return devices, p0, p1


@app.cell
def __(devices, p0, p1, plotter):
    plotter.capacity_plot(p0, p1, devices)[0]
    return


@app.cell
def __(p0, p1, problem):
    layer = problem["problem"].layer
    y0 = layer(**p0)
    y1 = layer(**p1)
    return layer, y0, y1


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
def __(devices, np, plt, y0):
    gens = devices[0]
    gen_power = y0.power[0][0]
    fuels = gens.fuel_type

    total_gen = np.sum(gen_power[fuels == "offwind", :], axis=0)

    plt.plot(total_gen)
    return fuels, gen_power, gens, total_gen


@app.cell
def __(layer, np, p1, plotter, plt, y1):
    def stackplot(ax, p1, layer, y1=None):
        if y1 is None:
            y1 = layer(**p1)

        devices = layer.devices

        # Plot total load
        loads = devices[1]
        total_load = -np.sum(loads.min_power * loads.nominal_capacity, axis=0)
        t = np.arange(total_load.size)
        ax.plot(t, total_load, color="black")

        # Stackplot generation
        gens = devices[0]
        gen_power = y1.power[0][0]
        fuels = gens.fuel_type

        gen_per_period = [
            np.sum(gen_power[fuels == f, :], axis=0) for f in plotter.FUEL_NAMES
        ]

        ax.stackplot(t, gen_per_period, labels=[f[:7] for f in plotter.FUEL_NAMES])

        # Plot battery output
        bats = devices[-2]
        bat_power = y1.power[-2][0]

        total_bat_power = np.sum(bat_power, axis=0)

        ax.fill_between(
            t,
            total_load,
            total_load - total_bat_power,
            color="yellow",
            alpha=0.5,
            label="battery",
        )

        # Tune figure
        ax.legend(fontsize=8, bbox_to_anchor=(1.2, 0.5), loc="center right")
        ax.set_xlim(np.min(t), np.max(t))

        return total_load


    _fig, _ax = plt.subplots(figsize=(6.5, 3))

    stackplot(_ax, p1, layer, y1)

    _fig.tight_layout()
    _fig
    return stackplot,


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
    # _prob = problem["stochastic_problem"]
    # _prob.forward(**result["parameters"], batch=[0])
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
def __():
    # plt.plot(all_dates[range(0, 8736, every)], renewable_curve)
    return


@app.cell
def __(pn):
    pn.generators[["carrier", "capital_cost"]].groupby("carrier").min().iloc[1:] / 1000
    return


@app.cell
def __(pn):
    solar_gens = pn.generators.carrier == "solar"
    df = pn.generators_t.p_max_pu

    solar_cols = [col for col in df.columns if "solar" in col]

    df[solar_cols].sum(axis=1).iloc[7:7+24].plot()
    return df, solar_cols, solar_gens


@app.cell
def __(pn):
    pn.lines["capital_cost"].mean() / 1000
    return


@app.cell
def __(devices, np, pn):
    np.all(pn.generators.carrier.values == devices[0].fuel_type)
    return


@app.cell
def __(devices):
    devices[0].num_devices
    return


@app.cell
def __(devices, np, plotter):
    {
        f: np.mean(devices[0].dynamic_capacity[devices[0].fuel_type == f, :])
        for f in plotter.FUEL_NAMES
    }
    return


@app.cell
def __(pn):
    pn.generators_t["p_max_pu"]
    return


if __name__ == "__main__":
    app.run()
