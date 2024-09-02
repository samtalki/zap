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
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )
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
    mo.md(r"""## Helper Functions""")
    return


@app.cell
def __(runner):
    def open_configs(path):
        return runner.expand_config(runner.load_config(path))
    return open_configs,


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
        df["num_nodes"] = [cfg["data"]["num_nodes"] for cfg in configs]

        # Extract runtime
        runtimes, data = zip(*[extract_runtime(cfg) for cfg in configs])

        df["mean_runtime"] = [np.mean(rt) for rt in runtimes]
        df["runtimes"] = runtimes

        return pd.DataFrame(df, index=index), data
    return build_runtime_table,


@app.cell
def __(Path, pickle, runner):
    def extract_runtime(config):
        path = Path(runner.get_results_path(config["id"], config["index"]))
        with open(path / "solver_data.pkl", "rb") as f:
            solver_data = pickle.load(f)

        runtimes = [[d["time"] for d in data] for data in solver_data]
        data = solver_data

        return runtimes, data
    return extract_runtime,


@app.cell
def __(plt):
    def plot_runtimes(
        df,
        fig=None,
        ax=None,
        compare="solver",
        x_index="scale_load",
        value="mean_runtime",
        labels=None,
    ):
        if fig is None:
            fig, ax = plt.subplots(figsize=(3, 2))

        keys = sorted(df[compare].unique())

        for k in keys:
            data = df[df[compare] == k]

            data = data[[x_index, value]]
            data = data.groupby(x_index).mean()
            data = data.sort_values(by=x_index)

            if labels is None:
                label = k
            else:
                label = labels[k]

            ax.plot(data.index, data[value], label=label, marker=".", ms=8)
            ax.set_xlabel(x_index)
            ax.set_ylabel(value)

        ax.legend()

        return fig, ax
    return plot_runtimes,


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Scaling""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Batch Size""")
    return


@app.cell
def __(build_runtime_table, open_configs):
    _configs = open_configs("./experiments/solve/config/scaling_hours_v01.yaml")
    df_hours, solver_data = build_runtime_table(_configs)

    df_hours["num_days"] = df_hours.hours_per_scenario / 24
    df_hours.sort_values(by=["hours_per_scenario", "solver"])
    return df_hours, solver_data


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Number of Nodes""")
    return


@app.cell
def __(build_runtime_table, open_configs):
    _configs = open_configs("./experiments/solve/config/scaling_devices_v02.yaml")
    df_nodes, _solver_data = build_runtime_table(_configs)

    df_nodes = df_nodes[df_nodes.num_nodes >= 500]

    df_nodes.sort_values(by=["num_nodes", "solver"])
    return df_nodes,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Combine Plots""")
    return


@app.cell
def __(Path, df_hours, df_nodes, plot_runtimes, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(7.5, 2.5))

    plot_runtimes(
        df_nodes,
        fig=_fig,
        ax=_axes[0],
        x_index="num_nodes",
        labels={"cvxpy": "Mosek", "admm": "ADMM"},
    )
    plot_runtimes(
        df_hours,
        fig=_fig,
        ax=_axes[1],
        x_index="hours_per_scenario",
        labels={"cvxpy": "Mosek", "admm": "ADMM"},
    )

    _axes[0].set_xlabel("Network Size")
    _axes[0].set_ylabel("Mean Runtime (s)")
    _axes[0].set_ylim(0.0, 15.0)

    _axes[1].get_legend().remove()
    _axes[1].set_xlabel("Time Horizon")
    _axes[1].set_ylabel("")
    _axes[1].set_yscale("log")
    _axes[1].set_ylim(1.0, 100.0)

    _fig.tight_layout()
    _fig.savefig(Path().home() / "figures/gpu/scaling_devices_hours.pdf")
    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Debug""")
    return


@app.cell
def __():
    # _index = 1
    # _path = Path(runner.get_results_path(configs[_index]["id"], _index))

    # with open(_path / "solver_data.pkl", "rb") as f:
    #     data = pickle.load(f)
    return


@app.cell
def __():
    # _c = data[1][0]
    # _c["time"]
    return


if __name__ == "__main__":
    app.run()
