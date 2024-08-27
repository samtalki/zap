import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


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
    seaborn.set_theme(style="whitegrid")
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
    config_path = "./experiments/solve/config/perf_v01.yaml"
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
def __(extract_runtime, np, pd):
    def build_runtime_table(configs):
        df = {}

        # Add config info
        # Index, solver, load_scale, hps full config reference
        index = [cfg["index"] for cfg in configs]
        df["solver"] = [cfg["solver"] for cfg in configs]
        df["scale_load"] = [cfg["data"]["args"]["scale_load"] for cfg in configs]
        df["hours_per_scenario"] = [
            cfg["parameters"]["hours_per_scenario"] for cfg in configs
        ]

        # Extract runtime
        runtimes = [extract_runtime(cfg) for cfg in configs]
        df["mean_runtime"] = [np.mean(rt) for rt in runtimes]
        df["runtimes"] = runtimes

        return pd.DataFrame(df, index=index)
    return build_runtime_table,


@app.cell
def __(build_runtime_table, configs):
    df = build_runtime_table(configs)
    df.sort_values(by=["hours_per_scenario", "solver"])
    return df,


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Plot Results""")
    return


@app.cell(hide_code=True)
def __(plt):
    def plot_runtimes(df, compare="solver", x_index="scale_load", value="mean_runtime"):
        fig, ax = plt.subplots(figsize=(7, 2.5))

        keys = sorted(df[compare].unique())

        for k in keys:
            data = df[df[compare] == k]

            data = data[[x_index, value]]
            data = data.groupby(x_index).mean()
            data = data.sort_values(by=x_index)

            ax.plot(data.index, data[value], label=k, marker=".", ms=8)
            ax.set_xlabel(x_index)
            ax.set_ylabel(value)

        ax.legend()

        return fig, ax
    return plot_runtimes,


@app.cell
def __(df, plot_runtimes):
    _fig, _ax = plot_runtimes(df, x_index="hours_per_scenario")
    _ax.set_yscale("log")
    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Debug""")
    return


@app.cell
def __(Path, configs, pickle, runner):
    _index = 1
    _path = Path(runner.get_results_path(configs[_index]["id"], _index))

    with open(_path / "solver_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data, f


@app.cell
def __():
    # _c = data[1][0]
    # _c["time"]
    return


if __name__ == "__main__":
    app.run()
