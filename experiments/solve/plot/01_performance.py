import marimo

__generated_with = "0.8.3"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    import importlib
    import pickle

    from pathlib import Path
    return Path, importlib, mo, np, pd, pickle, torch


@app.cell
def __():
    import zap
    return zap,


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn
    return plt, seaborn


@app.cell
def __(importlib):
    from experiments.solve import runner
    _ = importlib.reload(runner)
    return runner,


@app.cell
def __(importlib):
    from experiments.solve import plotter
    _ = importlib.reload(plotter)
    return plotter,


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Open Configs""")
    return


@app.cell
def __(runner):
    config_path = "./experiments/solve/config/test_v01.yaml"
    configs = runner.expand_config(runner.load_config(config_path))
    return config_path, configs


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Open Data and Extract Runtimes""")
    return


@app.cell
def __(Path, pickle, runner):
    def extract_runtime(config):
        path = Path(runner.get_results_path(config["id"], config["index"]))
        with open(path / "solver_data.pkl", "rb") as f:
            solver_data = pickle.load(f)

        return [[d["time"] for d in data] for data in solver_data]
    return extract_runtime,


@app.cell
def __():
    def build_runtime_table(configs):

        # Extract runtime

        # Add config info
        # Index, solver, load_scale, hps full config reference

        # Build a row in the table
        pass
    return build_runtime_table,


@app.cell
def __():
    # solver_data = []

    # for index, cfg in enumerate(configs):
    #     path = Path(runner.get_results_path(cfg["id"], index))
    #     with open(path / "solver_data.pkl", "rb") as f:
    #         solver_data += [pickle.load(f)]
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Plot Results""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Debug""")
    return


@app.cell
def __(configs, extract_runtime):
    extract_runtime(configs[0])
    return


@app.cell
def __(Path, configs, pickle, runner):
    _index = 1
    _path = Path(runner.get_results_path(configs[_index]["id"], _index))

    with open(_path / "solver_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data, f


@app.cell
def __(data):
    _c = data[1][0]
    _c["time"]
    return


if __name__ == "__main__":
    app.run()
