import marimo

__generated_with = "0.7.9"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import torch
    import importlib

    from pathlib import Path
    return Path, importlib, mo, np, torch


@app.cell
def __():
    import zap
    return zap,


@app.cell
def __(importlib):
    from experiments import runner
    _ = importlib.reload(runner)
    return runner,


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_theme(style="darkgrid")
    return plt, seaborn


@app.cell
def __(runner):
    # Load config data
    config = runner.load_config("experiments/config/demo_small_v01.yaml")
    return config,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
