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
    def build_runtime_table(configs, skip_missing=False):
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
        df["num_contingencies"] = [cfg.get("num_contingencies", 0) for cfg in configs]

        # Extract runtime
        runtimes, primal_residuals, dual_residuals, total_resids, data = zip(*[extract_runtime(cfg, skip_missing=skip_missing) for cfg in configs])

        df["median_runtime"] = [np.median(rt) for rt in runtimes]
        df["mean_runtime"] = [np.mean(rt) for rt in runtimes]
        df["runtimes"] = runtimes
        
        df["primal_residuals"] = primal_residuals
        df["dual_residuals"] = dual_residuals
        df["total_residuals"] = total_resids

        df["median_resid"] = [None if r is None else np.median(r) for r in total_resids]
        df["upper_resid"] = [None if r is None else np.max(r) for r in total_resids]

        df = pd.DataFrame(df, index=index)

        if skip_missing:
            df = df[df.mean_runtime >= 0.0]

        return df, data
    return build_runtime_table,


@app.cell
def __(np):
    def get_primal_resid_scaled(data):
        h = data["history"]
        root_n = np.sqrt(data["num_ac_terminals"]) + np.sqrt(data["num_dc_terminals"])

        return np.sqrt(h.power[-1] ** 2 + h.phase[-1] ** 2) / root_n

    def get_dual_resid_scaled(data):
        h = data["history"]
        root_n = np.sqrt(data["num_ac_terminals"]) + np.sqrt(data["num_dc_terminals"])

        return np.sqrt(h.dual_power[-1] ** 2 + h.dual_phase[-1] ** 2) / root_n
    return get_dual_resid_scaled, get_primal_resid_scaled


@app.cell
def __(
    Path,
    get_dual_resid_scaled,
    get_primal_resid_scaled,
    np,
    pickle,
    runner,
):
    def extract_runtime(config, skip_missing=False):
        path = Path(runner.get_results_path(config["id"], config["index"])) / "solver_data.pkl"

        if skip_missing and (not path.exists()):
            return [-1.0], None, None, None, None

        with open(path, "rb") as f:
            solver_data = pickle.load(f)

        runtimes = [[d["time"] for d in data] for data in solver_data]

        if config["solver"] == "admm":
            primal_residuals = [[get_primal_resid_scaled(d) for d in data] for data in solver_data]
            dual_residuals = [[get_dual_resid_scaled(d) for d in data] for data in solver_data]
            total_resids = [[np.maximum(rp, rd) for rp, rd in zip(primals, duals)] for primals, duals in zip(primal_residuals, dual_residuals)]
            
        else:
            primal_residuals = None
            dual_residuals = None
            total_resids = None
            
        data = solver_data

        return runtimes, primal_residuals, dual_residuals, total_resids, data
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
    mo.md(r"""### Time Horizon""")
    return


@app.cell
def __(solver_data):
    solver_data[0][0][0]
    return


@app.cell
def __(build_runtime_table, open_configs):
    _configs = open_configs("./experiments/solve/config/scaling_hours_v03.yaml")
    df_hours, solver_data = build_runtime_table(_configs)

    df_hours["num_days"] = df_hours.hours_per_scenario / 24
    df_hours.sort_values(by=["median_resid", "hours_per_scenario", "solver"])

    df_hours.sort_values(by=["upper_resid"], ascending=False)
    return df_hours, solver_data


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Number of Nodes""")
    return


@app.cell
def __(build_runtime_table, open_configs):
    _configs = open_configs("./experiments/solve/config/scaling_devices_v03.yaml")
    df_nodes, _solver_data = build_runtime_table(_configs)

    df_nodes = df_nodes[df_nodes.num_nodes >= 500]

    df_nodes.sort_values(by=["num_nodes", "solver"])
    df_nodes.sort_values(by=["upper_resid"], ascending=False)
    return df_nodes,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Contingencies""")
    return


@app.cell
def __(build_runtime_table, np, open_configs, pd):
    _configs = open_configs("./experiments/solve/config/scaling_cont_v04.yaml")
    df_cont, _solver_data = build_runtime_table(_configs, skip_missing=False)
    df_cont_big, _solver_data = build_runtime_table(
        open_configs("./experiments/solve/config/scaling_cont_big_v02.yaml")
    )
    df_cont_base, _solver_data = build_runtime_table(
        open_configs("./experiments/solve/config/scaling_cont_big_base_v01.yaml"),
        skip_missing=True,
    )


    df_cont["num_contingencies"] = np.minimum(df_cont["num_contingencies"], 1158) + 1
    df_cont_big["num_contingencies"] = np.minimum(df_cont_big["num_contingencies"], 1158) + 1
    df_cont_base["num_contingencies"] = np.minimum(df_cont_base["num_contingencies"], 1158) + 1


    df_cont = pd.concat([df_cont, df_cont_big, df_cont_base])


    df_cont.sort_values(by=["upper_resid"], ascending=False)
    return df_cont, df_cont_base, df_cont_big


@app.cell
def __(df_cont, df_hours, df_nodes, np):
    _median = np.median([df_hours["median_resid"].median(), df_nodes["median_resid"].median(), df_cont["median_resid"].median()])
    _max = np.max([df_hours["upper_resid"].max(), df_nodes["upper_resid"].max(), df_cont["upper_resid"].max()])

    print(f"Median Residual: {_median}")
    print(f"Max Residual: {_max}")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Combine Plots""")
    return


@app.cell
def __(Path, df_cont, df_hours, df_nodes, plot_runtimes, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(6.5, 2.5))

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
    plot_runtimes(
        df_cont,
        fig=_fig,
        ax=_axes[2],
        x_index="num_contingencies",
        labels={"cvxpy": "Mosek", "admm": "ADMM"},
    )

    _axes[0].set_xlabel("Network Size")
    _axes[0].set_ylabel("Mean Runtime (s)")
    _axes[0].set_ylim(0.0, 15.0)
    _axes[0].legend(loc="upper left")

    _axes[1].get_legend().remove()
    _axes[1].set_xlabel("Time Horizon")
    _axes[1].set_ylabel("")
    _axes[1].set_yscale("log")
    _axes[1].set_ylim(1.0, 1000.0)

    _axes[2].set_ylabel("")
    _axes[2].set_yscale("log")
    _axes[2].set_xscale("log")
    _axes[2].get_legend().remove()
    _axes[2].set_xlabel("Contingencies")

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
