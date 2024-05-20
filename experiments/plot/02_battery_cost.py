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
    import pandas as pd
    import json
    return json, pd, plt, seaborn


@app.cell
def __(importlib):
    from experiments import runner
    _ = importlib.reload(runner)
    return runner,


@app.cell
def __(Path):
    FIG_DIR = Path().home() / "figures/zap"
    return FIG_DIR,


@app.cell
def __(mo):
    mo.md("## Initial Setup")
    return


@app.cell
def __():
    config_name = "cost_battery_v03"
    return config_name,


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
def __(config_name, config_version, initial_params, model_iter, runner):
    if model_iter == 0:
        model_state = initial_params

    else:
        model_state = runner.load_model(
            f"{config_name}/{config_version:03d}/model_{model_iter:05d}"
        )
    return model_state,


@app.cell
def __(problem):
    initial_params = problem.initialize_parameters(None)
    return initial_params,


@app.cell
def __(mo):
    mo.md("## Summary Plots")
    return


@app.cell
def __(mo):
    slider = mo.ui.slider(0.1, 2.0, step=0.1)
    return slider,


@app.cell
def __(job_data, slider):
    config_version = job_data[job_data.battery_cost_scale == slider.value]["index"].iloc[0]
    return config_version,


@app.cell
def __(config_version):
    config_version
    return


@app.cell
def __():
    model_iter = 300
    return model_iter,


@app.cell
def __(devices, initial_params, model_state, plotter):
    plotter.capacity_plot(initial_params, model_state, devices)[0]
    return


@app.cell
def __(mo, slider):
    mo.md(f"""Battery Cost: {slider} {100 * slider.value}% reference value""")
    return


@app.cell
def __(config, data, runner):
    problem_data = runner.setup_problem(**data, **config["problem"])
    problem = problem_data["stochastic_problem"].subproblems[0]
    return problem, problem_data


@app.cell(hide_code=True)
def __(model_state, plotter, problem, slider, y_model):
    fig, ax = plotter.stackplot(model_state, problem.layer, y_model)
    _bat_cost = 100 * slider.value
    ax.set_title(f"Battery Cost: {_bat_cost}%")
    fig.tight_layout()

    fig
    return ax, fig


@app.cell
def __(model_state, problem):
    print("System Cost:", problem(**model_state))
    print("Investment Cost:", problem.inv_cost)
    print("Operation Cost:", problem.op_cost)
    y_model = problem.state
    return y_model,


@app.cell
def __(mo):
    mo.md("## Plot Expansions")
    return


@app.cell
def __():
    import wandb
    api = wandb.Api()
    return api, wandb


@app.cell
def __(importlib):
    from experiments import wb
    _ = importlib.reload(wb)
    return wb,


@app.cell
def __(api, wb):
    wandb_data = wb.get_wandb_data(api)
    return wandb_data,


@app.cell
def __(config_name, wandb_data):
    job_data = wandb_data[wandb_data.id == config_name]
    job_data = job_data[job_data.initial_state != "initial"]
    job_data = job_data.sort_values(by="battery_cost_scale")
    return job_data,


@app.cell
def __(EXPANSION_COSTS, config_name, job_data, model_iter, runner):
    models = []
    for e in EXPANSION_COSTS:
        model_id = job_data[job_data.battery_cost_scale == e]["index"].iloc[0]
        models += [
            runner.load_model(
                f"{config_name}/{model_id:03d}/model_{model_iter:05d}"
            )
        ]
    return e, model_id, models


@app.cell
def __():
    EXPANSION_COSTS = [0.5, 1.0, 2.0]
    return EXPANSION_COSTS,


@app.cell
def __(EXPANSION_COSTS, devices, initial_params, models, plotter, plt):
    def final_expansions_plot(fig=None, axes=None):
        if fig is None:
            fig, axes = plt.subplots(
                1, 4, figsize=(7, 2.5), width_ratios=[1, 1, 1, 10]
            )

        plotter.capacity_plot(
            initial_params,
            models,
            devices,
            fig=fig,
            axes=axes,
            label=[f"{int(100*c)}%" for c in EXPANSION_COSTS],
        )

        axes[1].set_ylim(0, 30)
        axes[2].set_ylim(0, 30)

        # fig.savefig(FIG_DIR / "battery_expansion.pdf")

        return fig, axes

    final_expansions_plot()[0]
    return final_expansions_plot,


@app.cell
def __(models, problem):
    ys = []
    for mod in models:
        problem(**mod)
        ys += [problem.state]
    return mod, ys


@app.cell
def __(importlib, seaborn):
    from experiments import plotter
    _ = importlib.reload(plotter)

    seaborn.set_theme(style="white", rc=plotter.SEABORN_RC)
    return plotter,


@app.cell
def __(models, plotter, plt, problem, ys):
    def double_fuel_plot(fig=None, axes=None):
        if fig is None:
            fig, axes = plt.subplots(1, 2, figsize=(7, 3))

        plotter.stackplot(
            models[0], problem.layer, ys[0], fig=fig, ax=axes[0], legend=False
        )
        plotter.stackplot(
            models[2], problem.layer, ys[2], fig=fig, ax=axes[1], legend=False
        )

        axes[0].set_ylabel("Power (MW)")
        [ax.set_xlabel("Hour") for ax in axes]
        [ax.set_xticks(range(0, 24, 6)) for ax in axes]
        axes[1].legend(fontsize=8, loc="lower center", ncol=3)
        fig.tight_layout()

        return fig


    double_fuel_plot()
    # _i = 1
    # plotter.stackplot(models[_i], problem.layer, ys[_i])
    return double_fuel_plot,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
