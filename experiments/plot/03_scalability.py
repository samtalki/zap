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
def __(importlib, seaborn):
    from experiments import plotter
    _ = importlib.reload(plotter)

    seaborn.set_theme(style="whitegrid", rc=plotter.SEABORN_RC)
    return plotter,


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
    config_name = "year_compare_v02"
    return config_name,


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
def __(api, config_name, wb):
    wbdf = wb.get_wandb_data(api)
    wbdf = wbdf[wbdf.id == config_name]
    return wbdf,


@app.cell
def __():
    wandb_cache = {}
    return wandb_cache,


@app.cell
def __(api, wandb_cache, wb, wbdf):
    settings = ["fixed", "sequential"]
    hashes = [
        wbdf[wbdf.batch_strategy == s].hash.values[0]
        for s in settings
    ]

    runs = [
        wb.get_run_history(api, h, cache=wandb_cache)
        for h in hashes
    ]
    return hashes, runs, settings


@app.cell
def __(FIG_DIR, np, plt, runs):
    def scalability_plot():
        fig, ax = plt.subplots(figsize=(7.5/2, 2))

        best = np.min(runs[1].full_loss)
        
        ax.plot(runs[0].full_loss / best, label="16 Days")
        ax.plot(runs[1].full_loss / best, label="360 Days")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective (360 Days)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(FIG_DIR / "scalability.pdf")

        return fig

    scalability_plot()
    return scalability_plot,


if __name__ == "__main__":
    app.run()
