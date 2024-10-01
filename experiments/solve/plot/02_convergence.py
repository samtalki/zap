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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
    mo.md(r"""## Open Configs and Build Table""")
    return


@app.cell
def __(runner):
    config_path = "./experiments/solve/config/converge_v04.yaml"
    configs = runner.expand_config(runner.load_config(config_path))
    return config_path, configs


@app.cell
def __(pd):
    def build_config_table(configs):
        df = {}

        # Add config info
        # Index, solver, load_scale, hps full config reference
        index = [cfg["index"] for cfg in configs]
        df["solver"] = [cfg["solver"] for cfg in configs]
        df["scale_load"] = [cfg["data"]["args"]["scale_load"] for cfg in configs]
        df["hours_per_scenario"] = [
            cfg["parameters"]["hours_per_scenario"] for cfg in configs
        ]
        df["config"] = [cfg for cfg in configs]

        return pd.DataFrame(df, index=index)
    return build_config_table,


@app.cell
def __(build_config_table, configs):
    df = build_config_table(configs)
    return df,


@app.cell
def __(df):
    df
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Open Convergence Histories""")
    return


@app.cell
def __():
    scale_load = 0.5
    hours_per_scenario = 24
    return hours_per_scenario, scale_load


@app.cell
def __(df, hours_per_scenario, scale_load):
    _df = df[(df.scale_load == scale_load) * (df.hours_per_scenario == hours_per_scenario)]
    baseline_index = _df[_df.solver == "cvxpy"].index[0]
    admm_index = _df[_df.solver == "admm"].index[0]
    return admm_index, baseline_index


@app.cell
def __(np):
    def get_objective_value(data, case_index=0, param_index=0):
        return np.sum([d["value"] for d in data["problem_data"]])
    return get_objective_value,


@app.cell
def __(baseline_index, configs, plotter):
    # Open baseline objective
    baseline_data = plotter.open_solver_data(configs[baseline_index])
    return baseline_data,


@app.cell
def __(admm_index, configs, plotter):
    # Open ADMM solver history
    admm_data = plotter.open_solver_data(configs[admm_index])
    return admm_data,


@app.cell
def __(admm_index, configs, pd, runner):
    arg_table = runner.runner.expand_config(configs[admm_index]["admm_args"], key="sweep")
    arg_table = pd.DataFrame(arg_table)
    # arg_table
    return arg_table,


@app.cell
def __():
    atol = 1.0e-4
    alpha = 1.0
    return alpha, atol


@app.cell
def __(alpha, arg_table, atol):
    _df = arg_table[(arg_table.atol == atol) * (arg_table.alpha == alpha)]
    assert _df.shape[0] == 1
    arg_index = _df.index[0]
    return arg_index,


@app.cell
def __(arg_index, baseline_data, plotter):
    case_index = 0
    admm_layer_index = plotter.reverse_index(case_index, arg_index, len(baseline_data[0]))
    return admm_layer_index, case_index


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Plot Convergence for Different Cases""")
    return


@app.cell
def __(
    admm_data,
    admm_layer_index,
    baseline_data,
    case_index,
    get_objective_value,
):
    print(len(admm_data[0][admm_layer_index]["history"].objective))
    print(admm_data[0][admm_layer_index]["history"].objective[0])
    print(admm_data[0][admm_layer_index]["history"].objective[-1])
    print(get_objective_value(baseline_data[0][case_index]))
    return


@app.cell
def __():
    # np.array(admm_data[0][admm_layer_index]["history"].power
    return


@app.cell
def __(
    Path,
    admm_data,
    admm_layer_index,
    baseline_data,
    case_index,
    get_objective_value,
    hours_per_scenario,
    plot_convergence,
):
    _fig, _axes = plot_convergence(
        admm_data[0][admm_layer_index],
        fstar=get_objective_value(baseline_data[0][case_index]),
        ylims=None,
    )

    _fig.savefig(Path().home() / f"figures/gpu/convergence_1_{hours_per_scenario}.pdf")

    _fig
    return


@app.cell
def __(np, plt):
    def plot_convergence(solver_data, fstar=1.0, ylims=(1e-3, 1e0)):
        hist = solver_data["history"]

        num_iters = len(hist.objective)
        root_n = np.sqrt(solver_data["num_ac_terminals"]) + np.sqrt(
            solver_data["num_dc_terminals"]
        )
        print(solver_data["primal_tol"] / root_n)

        total_primal = np.sqrt(np.power(hist.power, 2) + np.power(hist.phase, 2)) / root_n
        total_dual = (
            np.sqrt(np.power(hist.dual_power, 2) + np.power(hist.dual_phase, 2)) / root_n
        )

        print(np.argwhere((total_primal <= 1e-4) * (total_dual <= 1e-4))[0])

        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

        ax = axes[0]
        ax.plot(total_primal, label="Primal")
        ax.plot(total_dual, label="Dual")

        x1, x2 = ax.get_xlim()
        ax.hlines(
            solver_data["primal_tol"] / root_n,
            color="black",
            ls="--",
            zorder=-100,
            xmin=x1,
            xmax=x2,
        )
        ax.set_xlim(x1, x2)

        ax.set_yscale("log")
        ax.set_title("Residuals")
        ax.set_xlabel("Iteration")
        ax.legend(framealpha=1)

        ax.patch.set_linewidth(1)
        ax.patch.set_edgecolor("black")

        ax = axes[1]
        ax.plot(np.abs(hist.objective[:] - fstar) / fstar, label="ADMM")
        ax.set_yscale("log")

        # x1, x2 = ax.get_xlim()
        # ax.hlines(
        #     fstar, color="black", ls="--", label="Optimal", zorder=-100, xmin=x1, xmax=x2
        # )
        # ax.set_xlim(x1, x2)

        # ax.legend()
        ax.set_title(r"$|f^{(i)} - f^*| \ / \ f^*$")
        ax.set_xlabel("Iteration")

        fig.tight_layout()
        return fig, axes
    return plot_convergence,


@app.cell(hide_code=True)
def __(np, plt):
    def old_plot_convergence(solver_data, fstar=1.0, ylims=(1e-3, 1e0)):
        hist = solver_data["history"]
        admm_num_iters = len(hist.objective)

        total_primal = np.sqrt(np.power(hist.power, 2) + np.power(hist.phase, 2))
        total_dual = np.sqrt(np.power(hist.dual_power, 2) + np.power(hist.dual_phase, 2))

        hline_settings = {"zorder": -100, "xmin": 0, "xmax": admm_num_iters}

        fig, axes = plt.subplots(2, 2, figsize=(7, 3.5))

        ax = axes[0][0]
        ax.hlines(solver_data["primal_tol"], color="black", **hline_settings)
        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.plot(total_primal, color="black", ls="dashed")
        ax.set_yscale("log")
        ax.set_title("primal residuals")
        if ylims is not None:
            ax.set_ylim(*ylims)

        ax = axes[0][1]
        ax.hlines(solver_data["dual_tol"], color="black", **hline_settings)
        ax.plot(hist.dual_power, label="power")
        ax.plot(hist.dual_phase, label="angle")
        ax.plot(total_dual, color="black", ls="dashed")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("dual residuals")
        if ylims is not None:
            ax.set_ylim(*ylims)

        ax = axes[1][0]
        ax.plot(np.abs(np.array(hist.objective) - fstar) / fstar)
        ax.set_yscale("log")
        ax.set_title("|f - f*| / f*")

        # ax = axes[1][1]
        # if len(hist.price_error) > 0:
        #     ax.plot(np.array(hist.price_error) / simple_result.prices.size)
        # ax.set_yscale("log")
        # ax.set_title("nu - nu*")

        fig.tight_layout()
        return fig
    return old_plot_convergence,


if __name__ == "__main__":
    app.run()
