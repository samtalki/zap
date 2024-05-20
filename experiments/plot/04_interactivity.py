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
    config_name = "warm_v06"
    return config_name,


@app.cell
def __(config_name, runner):
    config_list = runner.expand_config(
        runner.load_config(f"experiments/config/{config_name}.yaml")
    )

    config = config_list[0]
    return config, config_list


@app.cell
def __(config, runner):
    data = runner.load_dataset(**config["data"])
    devices = data["devices"]
    return data, devices


@app.cell
def __(problem):
    initial_params = problem.initialize_parameters(None)
    return initial_params,


@app.cell
def __():
    model_iter = 250
    return model_iter,


@app.cell
def __(config, data, runner):
    problem_data = runner.setup_problem(**data, **config["problem"])
    problem = problem_data["stochastic_problem"].subproblems[0]
    return problem, problem_data


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
    job_data = job_data.sort_values(by="emissions_weight")
    return job_data,


@app.cell
def __(mo):
    mo.md("## Plot Fuel Mix")
    return


@app.cell
def __(importlib, seaborn):
    from experiments import plotter
    _ = importlib.reload(plotter)

    seaborn.set_theme(style="white", rc=plotter.SEABORN_RC)
    return plotter,


@app.cell
def __(single_fuel_plot):
    single_fuel_plot(weight=150.0, savename="fuel_mix", title="", height=1.5)
    return


@app.cell
def __(single_fuel_plot):
    single_fuel_plot(weight=300.0, savename="fuel_mix", title="", height=1.5)
    return


@app.cell
def __(single_fuel_plot):
    single_fuel_plot(
        weight=300.0,
        savename="fuel_mix_legend",
        title="",
        legend=True,
    )
    return


@app.cell
def __(
    FIG_DIR,
    config_name,
    job_data,
    model_iter,
    plotter,
    plt,
    problem,
    runner,
):
    def single_fuel_plot(
        weight, fig=None, axes=None, savename=None, title=None, legend=False, height=2.0
    ):
        fig, ax = plt.subplots(figsize=(3.75 - 0.1, height))

        config_version = job_data[job_data.emissions_weight == weight]["index"].iloc[0]
        model_state = runner.load_model(
            f"{config_name}/{config_version:03d}/model_{model_iter:05d}"
        )
        problem(**model_state)
        y_model = problem.state

        plotter.stackplot(
            model_state, problem.layer, y_model, fig=fig, ax=ax, legend=False
        )

        ax.set_ylabel("Power (GW)")
        ax.set_xlabel("Hour")
        ax.set_xticks(range(0, 24, 6))
        ax.set_title(title)

        if legend:
            legend = ax.legend(ncol=2, loc="lower center", fontsize=6)
            legend.get_frame().set_alpha(None)

        fig.tight_layout()
        if savename is not None:
            fig.savefig(FIG_DIR / f"{savename}_carbon_{int(weight):03d}.pdf")

        return fig
    return single_fuel_plot,


if __name__ == "__main__":
    app.run()
