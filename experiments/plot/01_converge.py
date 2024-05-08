import marimo

__generated_with = "0.4.7"
app = marimo.App()


@app.cell
def __():
    # import marimo as mo
    import numpy as np
    import torch
    import importlib

    from pathlib import Path

    import zap
    return Path, importlib, np, torch, zap


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn
    import pandas as pd
    import json
    return json, pd, plt, seaborn


@app.cell
def __(importlib):
    from experiments import runner
    _ = importlib.reload(runner)
    return runner,


@app.cell
def __(config_name, config_version, runner):
    config_list = runner.expand_config(
        runner.load_config(f"experiments/config/{config_name}.yaml")
    )

    config = config_list[config_version]
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
def __(config_version, initial_params, model_iter, runner):
    if model_iter == 0:
        model_state = initial_params

    else:
        model_state = runner.load_model(
            f"cost_battery_v01/{config_version:03d}/model_{model_iter:05d}"
        )
    return model_state,


@app.cell
def __(problem):
    initial_params = problem.initialize_parameters(None)
    return initial_params,


@app.cell
def __(importlib, seaborn):
    from experiments import plotter
    _ = importlib.reload(plotter)

    seaborn.set_theme(style="whitegrid", rc=plotter.SEABORN_RC)
    return plotter,


@app.cell
def __():
    config_name = "cost_battery_v01"
    return config_name,


@app.cell
def __(config_version, mo, slider):
    mo.md(f"""Battery Cost: {slider} {100 + 25 * (config_version - 2)}% reference value""")
    return


@app.cell
def __(mo):
    slider = mo.ui.slider(0, 6)
    return slider,


@app.cell
def __(slider):
    config_version = slider.value
    return config_version,


@app.cell
def __():
    model_iter = 100
    return model_iter,


@app.cell
def __(devices, initial_params, model_state, plotter):
    plotter.capacity_plot(initial_params, model_state, devices)[0]
    return


@app.cell
def __(config_version, model_state, plotter, problem, y_model):
    fig, ax = plotter.stackplot(model_state, problem.layer, y_model)
    _bat_cost = 100 + 25 * (config_version - 2)
    ax.set_title(f"Battery Cost: {_bat_cost}%")
    fig.tight_layout()

    fig
    return ax, fig


@app.cell
def __():
    y_cache = {}
    return y_cache,


@app.cell
def __(config_version, model_state, problem, y_cache):
    # _prob = problem
    # print("System Cost:", _prob(**model_state, batch=[0]))
    # print("Investment Cost:", _prob.inv_cost)
    # print("Operation Cost:", _prob.op_cost)
    if config_version not in y_cache:
        _prob = problem
        _prob(**model_state, batch=[0])
        y_cache[config_version] = _prob.state
        
    y_model = y_cache[config_version]
    return y_model,


@app.cell
def __(mo):
    mo.md("## Plot Convergence Rate")
    return


@app.cell
def __(np):
    EMISSIONS_WEIGHTS = np.arange(100.0, 300.0 + 1, 20.0)
    return EMISSIONS_WEIGHTS,


@app.cell
def __():
    REFERENCE_WEIGHT = 200.0
    return REFERENCE_WEIGHT,


@app.cell
def __():
    convergence_config = "warm_v05"
    return convergence_config,


@app.cell
def __():
    # convergence_settings = runner.expand_config(
    #     runner.load_config(f"experiments/config/{convergence_config}.yaml")
    # )
    return


@app.cell
def __():
    import wandb
    return wandb,


@app.cell
def __(wandb):
    api = wandb.Api()
    return api,


@app.cell
def __(api):
    runs = api.runs("s3l/zap")
    return runs,


@app.cell
def __(pd, runs):
    _wd = {
        "name": [],
        "id": [],
        "index": [],
        "emissions_weight": [],
        "initial_state": [],
        "hash": [],
    }

    for r in runs:
        _wd["name"].append(r.name)
        _wd["id"].append(r.config.get("id", ""))
        _wd["index"].append(r.config.get("index", -1))
        _wd["hash"].append(r.id)

        _wd["emissions_weight"].append(
            float(r.config["problem"]["emissions_weight"])
        )
        _wd["initial_state"].append(r.config["optimizer"]["initial_state"])

    wandb_data = pd.DataFrame(_wd)
    return r, wandb_data


@app.cell
def __(convergence_config, wandb_data):
    job_data = wandb_data[wandb_data["id"] == convergence_config].reset_index()
    return job_data,


@app.cell
def __(job_data):
    job_data
    return


@app.cell
def __(api):
    wandb_cache = {}

    def get_run_history(hash, cache=wandb_cache):
        if hash not in wandb_cache:
            run = api.run(f"s3l/zap/{hash}")
            wandb_cache[hash] = run.history()

        return wandb_cache[hash]
    return get_run_history, wandb_cache


@app.cell
def __(CTOL, LAG, np):
    def has_converged(loss, tolerance=CTOL, lag=LAG):
        best_loss = [np.min(loss[:t+1]) for t in range(loss.size)]

        for iter in range(loss.size):
            if iter < lag:
                continue

            current_loss = best_loss[iter]
            last_loss = best_loss[iter-lag]

            if current_loss > (1 - tolerance) * last_loss:
                return iter

        return loss.size - 1
    return has_converged,


@app.cell
def __(get_run_history, job_data):
    # Load loss curve for each run
    _history = [get_run_history(h) for h in job_data.hash]
    losses = [h["rolling_loss"] for h in _history]
    return losses,


@app.cell
def __(job_data):
    job_data[job_data.emissions_weight == 100.0].index
    return


@app.cell
def __(REFERENCE_WEIGHT, has_converged, job_data, losses, np, plt):
    _carbon = 160.0
    _df = job_data[job_data.emissions_weight == _carbon]

    _fig = plt.figure(figsize=(6, 2.5))
    plt.title(f"emissions weight: {_carbon} (warm start weight: {REFERENCE_WEIGHT})")

    for _i in _df.index:
        _loss = losses[_i]
        _bl = [np.min(_loss[: t + 1]) for t in range(_loss.size)]
        _cvg_iter = has_converged(_loss)

        print(
            f"Converged in {_cvg_iter} iterations with subopt: {100 * _bl[_cvg_iter] / _bl[-1] - 100.0:.2f}%"
        )

        # plt.plot(_bl)
        plt.plot(_loss)
        plt.scatter([_cvg_iter], [_bl[_cvg_iter]], c="red", zorder=1000)


    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.ylim(6500, 12_000)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def __(has_converged, job_data, losses):
    _df = job_data.copy()
    _df["converged"] = [has_converged(l) for l in losses]

    cold_converge_times = (
        _df[_df.initial_state == "initial"]
        .sort_values(by="emissions_weight")
        .converged.values
    )
    warm_converge_times = (
        _df[_df.initial_state != "initial"]
        .sort_values(by="emissions_weight")
        .converged.values
    )
    return cold_converge_times, warm_converge_times


@app.cell
def __(cold_converge_times, np, warm_converge_times):
    print("Mean Cold Convergence:", np.median(cold_converge_times))
    print("Mean Warm Convergence:", np.median(warm_converge_times))
    return


@app.cell
def __(has_converged, losses, np):
    subopts = [np.min(l[:has_converged(l)+1]) / np.min(l) - 1.0 for l in losses]
    print(f"Worst Suboptimality: {100 * np.max(subopts):.2f}%")
    print(f"Mean Suboptimality: {100 * np.mean(subopts):.2f}%")
    return subopts,


@app.cell
def __():
    LAG = 10
    CTOL = 0.001
    return CTOL, LAG


@app.cell
def __(
    EMISSIONS_WEIGHTS,
    REFERENCE_WEIGHT,
    cold_converge_times,
    plt,
    warm_converge_times,
):
    def plot_convergence_results():
        speedups = cold_converge_times / warm_converge_times
        perturb = (EMISSIONS_WEIGHTS - REFERENCE_WEIGHT) / REFERENCE_WEIGHT

        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        ax.set_xlabel("Perturbation Size")
        ax.set_ylabel("Speedup")

        # Absolute time
        # ax.plot(perturb, cold_converge_times, marker=".")
        # ax.plot(perturb, warm_converge_times, marker=".")
        # ax.set_ylim(0, 250)

        # Speedup
        ax.plot(perturb, speedups)
        ax.set_ylim(0, ax.get_ylim()[1])
        fig.tight_layout()

        return fig

    plot_convergence_results()
    return plot_convergence_results,


@app.cell
def __(runner):
    _cfg = runner.expand_config(
        runner.load_config(f"experiments/config/cost_reconductor_v02.yaml")
    )

    _cfg = _cfg[0]
    _data = runner.load_dataset(**_cfg["data"])
    _devs = _data["devices"]

    lines = _devs[3]
    return lines,


@app.cell
def __(lines, np):
    _x = np.copy(lines.nominal_capacity)

    b = lines.nominal_capacity
    c = lines.capital_cost
    _r = lines.reconductoring_cost
    alpha = lines.reconductoring_threshold

    z = b * (_r + c * alpha - _r * alpha)

    def foo(shift):    
        return np.sum(
            np.maximum(
                np.multiply(_r, (_x - b)),
                np.multiply(c, _x) - z,
            )
        )

        return np.multiply(c, _x) - z


    # _t = np.arange(b, 3.0, 0.01)
    # _y = [foo(ti) for ti in _t]
    # plt.plot(_t, _y)

    # print(_x[223], b[223], c[223], _r[223], alpha[223])
    # print(c[223] * _x[223], z[223])
    # np.multiply(c[223], _x[223])

    print(foo(_x))
    return alpha, b, c, foo, z


if __name__ == "__main__":
    app.run()
