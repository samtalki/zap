import marimo

__generated_with = "0.8.14"
app = marimo.App()


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
    import matplotlib.pyplot as plt
    import seaborn
    seaborn.set_theme(
        style="whitegrid",
        palette="bright",
        rc={
            "axes.edgecolor": "0.15",
            "axes.linewidth": 1.25,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
    )
    return plt, seaborn


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

        _wd["emissions_weight"].append(
            float(r.config["problem"]["emissions_weight"])
        )
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
            wandb_cache[hash] = run.history()

        return wandb_cache[hash]
    return get_run_history, wandb_cache


@app.cell
def __(get_run_history, job_data):
    history = get_run_history(job_data.hash[0])
    return history,


@app.cell
def __(history):
    history.admm_iteration.diff()
    return


@app.cell
def __(Path, history, plt):
    def loss_plot(num_iter=100):
        
        loss = history.loss[:num_iter]
        admm_iter = history.admm_iteration.diff()[1:num_iter+1]

        fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5))

        # Plot loss
        axes[0].plot(loss / 1e3)
        
        # Plot ADMM iterations per iter
        axes[1].plot(admm_iter)
        axes[1].set_ylim(0, 1500)
        

        axes[1].set_xlabel("Gradient Descent Steps")
        axes[0].set_ylabel("Loss")
        axes[1].set_ylabel("Iterations per Step")

        fig.align_ylabels(axes)
        fig.tight_layout()
        fig.savefig(Path.home() / "figures/gpu_admm_planning.pdf")
        
        return fig


    loss_plot()
    return loss_plot,


if __name__ == "__main__":
    app.run()
