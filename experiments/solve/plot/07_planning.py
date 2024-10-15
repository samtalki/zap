import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import pandas as pd
    import scipy.sparse as sp
    import wandb as wb

    import torch
    import importlib
    import pypsa
    import datetime as dt

    from copy import deepcopy
    from pathlib import Path
    return (
        Path,
        cp,
        deepcopy,
        dt,
        importlib,
        mo,
        np,
        pd,
        pypsa,
        sp,
        torch,
        wb,
    )


@app.cell
def __():
    import zap
    return zap,


@app.cell
def __():
    from experiments.plan import plotter
    from experiments.plan import runner
    return plotter, runner


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn
    from experiments.solve import plotter as formatter
    return formatter, plt, seaborn


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Load Wandb File""")
    return


@app.cell
def __(wb):
    api = wb.Api()
    return api,


@app.cell
def __(api):
    runs = api.runs("s3l/zap")
    return runs,


@app.cell(hide_code=True)
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

        _wd["emissions_weight"].append(float(r.config["problem"]["emissions_weight"]))
        _wd["initial_state"].append(r.config["optimizer"]["initial_state"])

    wandb_data = pd.DataFrame(_wd)
    return r, wandb_data


@app.cell
def __(wandb_data):
    wandb_data
    return


@app.cell
def __(wandb_data):
    job_data = wandb_data[wandb_data["name"] == "dashing-sun-633"].reset_index()
    return job_data,


@app.cell
def __(api):
    wandb_cache = {}


    def get_run_history(hash, cache=wandb_cache):
        if hash not in wandb_cache:
            run = api.run(f"s3l/zap/{hash}")
            wandb_cache[hash] = run.history(samples=2000)

        return wandb_cache[hash]
    return get_run_history, wandb_cache


@app.cell
def __(get_run_history, job_data):
    history = get_run_history(job_data.hash[0])
    return history,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Load Solution""")
    return


@app.cell
def __(job_data):
    config_name = job_data.id[0]
    config_id = job_data.index[0]
    return config_id, config_name


@app.cell
def __(config_id, config_name, runner):
    config = runner.expand_config(
        runner.load_config(f"experiments/plan/config/{config_name}.yaml")
    )[config_id]
    data = runner.load_dataset(**config["data"])
    devices = data["devices"]
    return config, data, devices


@app.cell
def __(config, data, runner):
    problem_data = runner.setup_problem(**data, **config["problem"])
    problem = problem_data["stochastic_problem"].subproblems[0]
    return problem, problem_data


@app.cell
def __(problem):
    initial_params = problem.initialize_parameters(None)
    return initial_params,


@app.cell
def __():
    model_iters = [10, 50, 100]
    return model_iters,


@app.cell
def __(config_id, config_name, model_iters, runner):
    model_states = [
        runner.load_model(f"{config_name}/{config_id:03d}/model_{i:05d}")
        for i in model_iters
    ]
    return model_states,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Plot Results""")
    return


@app.cell(hide_code=True)
def __(history, model_iters, np, plt):
    def loss_plot(num_iter=np.max(model_iters), figsize=(6.5, 2)):
        loss = history.loss[:num_iter]
        admm_iter = history.admm_iteration.diff()[1 : num_iter + 1]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        marks = np.array(model_iters) - 1

        # Plot loss
        axes[0].plot(loss / 1e3)
        axes[0].scatter(
            marks,
            history.loss[marks] / 1e3,
            c=["C1", "C2", "C3"],
            zorder=100,
        )

        # Plot ADMM iterations per iter
        axes[1].plot(admm_iter)
        axes[1].set_ylim(0, 1100)

        axes[0].set_xlabel("Gradient Descent Steps")
        axes[0].set_ylabel("Loss")

        axes[1].set_xlabel("Gradient Descent Steps")
        axes[1].set_ylabel("Iterations per Step")

        # fig.align_ylabels(axes)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.5)

        return fig
    return loss_plot,


@app.cell
def __(Path, formatter, loss_plot):
    formatter.set_full_style()
    _fig = loss_plot(figsize=(formatter.FIGWIDTH_FULL, 2))
    _fig.savefig(Path.home() / "figures/gpu/admm_planning_fig_full.eps")

    _fig
    return


@app.cell
def __(Path, formatter, loss_plot):
    formatter.set_small_style()
    _fig = loss_plot(figsize=(formatter.FIGWIDTH_SMALL, 1.5))
    _fig.savefig(Path.home() / "figures/gpu/admm_planning_fig_small.eps")
    _fig.savefig(Path.home() / "figures/gpu/Fig6.eps")

    _fig
    return


@app.cell
def __(devices, initial_params, model_iters, model_states, plotter, plt):
    def capacity_plot(figsize=(6.5, 2.5)):
        fig, axes = plt.subplots(1, 4, figsize=figsize, width_ratios=[1, 1, 1, 6])

        labels = [f"{i} Steps" for i in model_iters]
        plotter.capacity_plot(
            initial_params,
            model_states,
            devices,
            fig=fig,
            axes=axes,
            label=labels,
            fuel_groups=[
                # "Nuclear",
                "Wind",
                "Solar",
                "Gas",
                "Hydro",
                "Coal",
            ],
        )

        axes[0].set_ylim(0, 8000)
        axes[1].set_ylim(0, 25)
        axes[2].set_ylim(0, 100)
        axes[3].set_ylim(0, 500)

        fig.tight_layout()

        return fig
    return capacity_plot,


@app.cell
def __(Path, capacity_plot, formatter):
    formatter.set_full_style()
    _fig = capacity_plot(figsize=(formatter.FIGWIDTH_FULL, 2.5))
    _fig.savefig(Path.home() / "figures/gpu/admm_planning_outcome_fig_full.eps")
    return


@app.cell
def __(Path, capacity_plot, formatter):
    formatter.set_small_style()
    _fig = capacity_plot(figsize=(formatter.FIGWIDTH_SMALL, 2))
    _fig.savefig(Path.home() / "figures/gpu/admm_planning_outcome_fig_small.eps")
    _fig.savefig(Path.home() / "figures/gpu/Fig7.eps")

    _fig
    return


if __name__ == "__main__":
    app.run()
