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
    configs = runner.expand_config(runner.load_config("./experiments/solve/config/warm_v02.yaml"))
    return configs,


@app.cell
def __(configs, pd):
    def build_config_table(path):
        df = {}

        # Add config info
        # Index, solver, load_scale, hps full config reference
        index = [cfg["index"] for cfg in configs]
        df["solver"] = [cfg["solver"] for cfg in configs]
        df["scale_load"] = [cfg["data"]["args"]["scale_load"] for cfg in configs]
        df["hours_per_scenario"] = [
            cfg["parameters"]["hours_per_scenario"] for cfg in configs
        ]
        df["warm_start"] = [cfg.get("warm_start", False) for cfg in configs]
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
def __(np):
    def get_objective_value(data, case_index=0, param_index=0):
        return np.sum([d["value"] for d in data["problem_data"]])
    return get_objective_value,


@app.cell
def __(configs, pd, plotter, runner):
    def load_case_data(df, scale_load=0.5, hours_per_scenario=24, warm_start=False):
        df = df[(df.scale_load == scale_load) * (df.hours_per_scenario == hours_per_scenario) * (df.warm_start == warm_start)]
        
        admm_index = df[df.solver == "admm"].index[0]
        admm_data = plotter.open_solver_data(configs[admm_index])

        arg_table = runner.runner.expand_config(configs[admm_index]["admm_args"], key="sweep")
        arg_table = pd.DataFrame(arg_table)

        return admm_data, arg_table
    return load_case_data,


@app.cell
def __(plotter):
    def select_case(data, arg_table, case_index, alpha=1.0):
        df = arg_table[(arg_table.alpha == alpha)]
        assert df.shape[0] == 1
        arg_index = df.index[0]
        
        num_cases = len(data[0]) // arg_table.shape[0]

        admm_layer_index = plotter.reverse_index(case_index, arg_index, num_cases)

        return data[0][admm_layer_index]
    return select_case,


@app.cell
def __(df, load_case_data):
    data_cold, args_cold = load_case_data(df, warm_start=False)
    return args_cold, data_cold


@app.cell
def __(df, load_case_data):
    data_warm, args_warm = load_case_data(df, warm_start=True)
    return args_warm, data_warm


@app.cell
def __(
    args_cold,
    args_warm,
    case_index,
    data_cold,
    data_warm,
    select_case,
):
    case_cold = select_case(data_cold, args_cold, case_index)
    case_warm = select_case(data_warm, args_warm, case_index)
    return case_cold, case_warm


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Plot Convergence for Different Cases""")
    return


@app.cell(hide_code=True)
def __(case_cold, case_warm):
    print(f"Number of Iterations: {len(case_cold['history'].objective)}")

    print(f"Cold Objective: {case_cold['history'].objective[-1]}")
    print(f"Cold Objective: {case_warm['history'].objective[-1]}")
    return


@app.cell
def __(case_cold, plot_convergence):
    _fig, _axes = plot_convergence(case_cold, ylims=None)
    _fig
    return


@app.cell
def __(case_warm, plot_convergence):
    _fig, _axes = plot_convergence(case_warm, ylims=None)
    _fig
    return


@app.cell
def __(np):
    def first_converged(data, unscaled_tol):
        root_n = np.sqrt(data["num_ac_terminals"]) + np.sqrt(data["num_dc_terminals"])
        history = data["history"]

        tol = unscaled_tol * root_n
        # print(unscaled_tol, tol)

        for i in range(len(history.objective)):
            r_primal = np.sqrt(history.power[i] ** 2 + history.phase[i] ** 2)
            r_dual = np.sqrt(history.dual_power[i] ** 2 + history.dual_phase[i] ** 2)

            if (r_primal < tol) and (r_dual < tol):
                return i

        return None
    return first_converged,


@app.cell
def __(first_converged):
    def get_convergence(tolerance_range, case):
        return [first_converged(case, tol) for tol in tolerance_range]
    return get_convergence,


@app.cell
def __():
    case_index = 1
    return case_index,


@app.cell
def __(case_cold, case_warm, first_converged):
    print(first_converged(case_cold, 1e-3))
    print(first_converged(case_warm, 1e-3))
    return


@app.cell
def __(Path, case_cold, case_warm, get_convergence, np, plt):
    tolerance_range = np.power(10.0, np.linspace(-2.0, -6.0, num=30, endpoint=True))
    #convergence_iters = [first_converged(data, hist, tol) for tol in tolerance_range]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.plot(tolerance_range, get_convergence(tolerance_range, case_cold), label="Zero Start")
    ax.plot(tolerance_range, get_convergence(tolerance_range, case_warm), label="Warm Start")

    ax.legend()
    ax.set_xlim(1.0e-6, 1.0e-1)
    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Accuracy")

    ax.set_yscale("log")
    ax.set_ylabel("Iterations")
    ax.set_ylim(10, 100_000)

    fig.tight_layout()
    fig.savefig(Path().home() / "figures/gpu/warm.pdf")

    fig
    return ax, fig, tolerance_range


@app.cell(hide_code=True)
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
        ax.legend()

        ax.patch.set_linewidth(1)
        ax.patch.set_edgecolor("black")

        ax = axes[1]
        ax.plot(hist.objective, label="ADMM")

        x1, x2 = ax.get_xlim()
        ax.hlines(
            fstar, color="black", ls="--", label="Optimal", zorder=-100, xmin=x1, xmax=x2
        )
        ax.set_xlim(x1, x2)

        ax.legend()
        ax.set_title("Objective Value")
        ax.set_xlabel("Iteration")

        fig.tight_layout()
        return fig, axes
    return plot_convergence,


if __name__ == "__main__":
    app.run()
