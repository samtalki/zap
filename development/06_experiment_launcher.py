import marimo

__generated_with = "0.7.1"
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

    seaborn.set_theme(
        style="whitegrid",
        rc={
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # "axes.ticksize": 8,
        },
    )
    return plt, seaborn


@app.cell
def __(importlib):
    from experiments.plan import runner

    _ = importlib.reload(runner)
    return runner,


@app.cell
def __():
    # devices[1].quadratic_cost = (1.0 / 500.0) * devices[1].linear_cost * np.ones_like(devices[1].linear_cost)
    return


@app.cell
def __(runner):
    config_list = runner.expand_config(
        runner.load_config("experiments/plan/config/test_gpu_v10.yaml")
    )

    config = config_list[0]
    return config, config_list


@app.cell
def __(config, runner):
    data = runner.load_dataset(**config["data"])
    return data,


@app.cell
def __(torch):
    torch.cuda.empty_cache()
    return


@app.cell
def __(config, data, runner):
    problem = runner.setup_problem(**data, **config["problem"], **config["layer"])
    return problem,


@app.cell
def __(config, problem, runner):
    relax = runner.solve_relaxed_problem(problem, **config["relaxation"])
    return relax,


@app.cell
def __(relax):
    if relax is not None:
        print("Solve time: ", relax["data"]["problem"].solver_stats.solve_time)
    return


@app.cell
def __(problem, relax):
    if relax is not None:
        _J = problem["problem"]
        print(_J(**relax["relaxed_parameters"]))
        print(relax["lower_bound"])
    return


@app.cell
def __(torch):
    torch.cuda.empty_cache()
    return


@app.cell
def __(config, problem, relax, runner):
    result = runner.solve_problem(problem, relax, config, **config["optimizer"])
    return result,


@app.cell(hide_code=True)
def __(np, plt):
    def plot_convergence(hist, eps_pd=None):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2))

        print(f"Primal Resid: {hist.power[-1] + hist.phase[-1]}")
        print(f"Dual Resid: {hist.dual_power[-1] + hist.dual_phase[-1]}")
        print(f"Objective: {hist.objective[-1]}")

        admm_num_iters = len(hist.power)

        ax = axes[0]
        if eps_pd is not None:
            ax.hlines(
                eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100
            )
        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title("primal residuals")

        ax = axes[1]
        if eps_pd is not None:
            ax.hlines(
                eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100
            )
        ax.plot(hist.dual_power, label="power")
        ax.plot(hist.dual_phase, label="angle")
        ax.set_yscale("log")
        # ax.legend(fontsize=8)
        ax.set_title("dual residuals")

        ax = axes[2]
        ax.plot(np.array(hist.objective))
        # ax.set_yscale("log")
        ax.set_title("f")

        # ax = axes[1][1]
        # if len(hist.price_error) > 0:
        #     ax.plot(np.array(hist.price_error) / simple_result.prices.size)
        # ax.set_yscale("log")
        # ax.set_title("nu - nu*")

        fig.tight_layout()
        return fig
    return plot_convergence,


@app.cell
def __(np, problem, result):
    result  # Force dependency

    sp = problem["stochastic_problem"].subproblems
    prob0 = problem["stochastic_problem"].subproblems[0]
    # prob1 = problem["stochastic_problem"].subproblems[1]

    layer0 = prob0.layer
    eps_pd = layer0.solver.atol * np.sqrt(layer0.solver.total_terminals)
    s0 = sp[-1].layer.state.copy()
    return eps_pd, layer0, prob0, s0, sp


@app.cell
def __(eps_pd, plot_convergence, prob0, sp):
    L = _L = sp[0].layer

    # rtol = 1.0e-3

    _L.solver.num_iterations = 1000
    _L.solver.rho_power = 0.1
    # _L.solver.rho_angle = _L.solver.rho_power
    # _L.solver.rtol = rtol
    # _L.solver.minimum_iterations = 100
    # _L.warm_start = False

    _L(**prob0.initialize_parameters(None), initial_state=None)

    # _L.solver.num_iterations = 1000
    # _L.solver.rho_power = 1.0
    # _L.solver.rho_angle = 1.0
    # _L.solver.rtol = 1.0e-3
    # _L.solver.minimum_iterations = 100
    # _L.warm_start = True

    plot_convergence(_L.history, eps_pd=eps_pd)
    return L,


@app.cell
def __(L, sp, torch):
    sp[0].layer.devices[1].operation_cost(L.state.power[1], None, None, la=torch)
    return


@app.cell
def __(problem):
    print(problem["problem"].layer.devices[0].capital_cost[:5])
    print(
        sum(
            [
                sp.layer.devices[0].capital_cost[:5]
                for sp in problem["stochastic_problem"].subproblems
            ]
        )
    )
    return


@app.cell
def __():
    # print(problem["problem"](**result["parameters"]))
    return


@app.cell
def __(result):
    result["history"]["loss"]
    return


@app.cell
def __(np, plt, result):
    _fig, _axes = plt.subplots(2, 1, figsize=(8, 3))

    _axes[0].plot(result["history"]["loss"])
    _axes[1].plot(result["history"]["grad_norm"])
    _axes[1].set_yscale("log")

    print(np.log10(result["history"]["grad_norm"]))
    print(np.min(result["history"]["loss"]))

    _fig
    return


@app.cell
def __(config, relax, result, runner):
    runner.save_results(relax, result, config)
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
    from experiments.plan import plotter

    _ = importlib.reload(plotter)
    return plotter,


@app.cell
def __(json, model_iter, np, problem, torch):
    with open(
        f"./data/results/base_v05/000/model_{model_iter:05d}.json", "r"
    ) as f:
        model_state = json.load(f)

    _ref_shapes = {
        k: v.shape
        for k, v in problem["problem"].initialize_parameters(None).items()
    }
    model_state = {
        k: torch.tensor(
            np.array(v).reshape(_ref_shapes[k]), device="cuda", dtype=torch.float32
        )
        for k, v in model_state.items()
    }
    return f, model_state


@app.cell
def __():
    # problem["problem"](**model_state)
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
    model_iter = 100
    return model_iter,


@app.cell
def __():
    # plotter.capacity_plot(p0, p1, devices)[0]
    return


@app.cell
def __():
    # plotter.stackplot(p1, layer, y1)[0]
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
