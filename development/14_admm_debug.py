import marimo

__generated_with = "0.6.17"
app = marimo.App(app_title="ADMM Debugger")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import pandas as pd
    import datetime as dt

    import torch
    import importlib
    import pypsa
    import json

    from copy import deepcopy
    return cp, deepcopy, dt, importlib, json, mo, np, pd, pypsa, torch


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


@app.cell(hide_code=True)
def __(np, plt):
    def plot_convergence(hist, admm, fstar=1.0):
        fig, axes = plt.subplots(2, 2, figsize=(7, 3.5))

        admm_num_iters = len(hist.objective)
        eps_pd = admm.total_tol  # * np.sqrt(admm.total_terminals)

        ax = axes[0][0]
        total_primal = np.sqrt(np.power(hist.power, 2) + np.power(hist.phase, 2))
        ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.plot(total_primal, color="black", ls="dashed")
        ax.set_yscale("log")
        ax.set_title("primal residuals")

        ax = axes[0][1]
        total_dual = np.sqrt(
            np.power(hist.dual_power, 2) + np.power(hist.dual_phase, 2)
        )
        ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.dual_power, label="power")
        ax.plot(hist.dual_phase, label="angle")
        ax.plot(total_dual, color="black", ls="dashed")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("dual residuals")

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
    return plot_convergence,


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"## Data")
    return


@app.cell
def __(importlib):
    from experiments import runner

    _ = importlib.reload(runner)
    return runner,


@app.cell
def __(runner):
    _config_list = runner.expand_config(
        runner.load_config("experiments/config/year_gpu_test_v01.yaml")
    )
    config = _config_list[0]
    return config,


@app.cell
def __(config, runner):
    data = runner.load_dataset(**config["data"])
    net, devices = data["net"], data["devices"]
    return data, devices, net


@app.cell
def __(config, data, runner):
    cpu_problem_data = runner.setup_problem(**data, **config["problem"])
    cpu_full_prob, cpu_stoch_prob = (
        cpu_problem_data["problem"],
        cpu_problem_data["stochastic_problem"],
    )
    return cpu_full_prob, cpu_problem_data, cpu_stoch_prob


@app.cell
def __(config, data, runner, torch):
    problem_data = runner.setup_problem(
        **data, **config["problem"], **config["layer"]
    )
    full_prob, stoch_prob = (
        problem_data["problem"],
        problem_data["stochastic_problem"],
    )
    torch.cuda.empty_cache()
    return full_prob, problem_data, stoch_prob


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Parameters")
    return


@app.cell
def __(cpu_problem_data, json, model_id, np):
    with open(
        f"./data/results/year_compare_v03/000/model_{model_id:05d}.json", "r"
    ) as f:
        model_state = json.load(f)

    _ref_shapes = {
        k: v.shape
        for k, v in cpu_problem_data["problem"].initialize_parameters(None).items()
    }

    model_state = {
        k: np.array(model_state[k]).reshape(shape)
        for k, shape in _ref_shapes.items()
    }
    return f, model_state


@app.cell
def __(J_cpu, model_state):
    theta0_cpu = J_cpu.layer.initialize_parameters()
    theta0_cpu = model_state
    return theta0_cpu,


@app.cell
def __(theta0_cpu, torch, zap):
    theta0_gpu = zap.util.torchify(theta0_cpu, machine="cuda", dtype=torch.float32)
    return theta0_gpu,


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"## Baseline (CVX)")
    return


@app.cell
def __(cpu_stoch_prob, problem_id):
    J_cpu = cpu_stoch_prob.subproblems[problem_id]
    return J_cpu,


@app.cell
def __(J_cpu, theta0_cpu):
    y0 = J_cpu.layer(**theta0_cpu)
    return y0,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## ADMM")
    return


@app.cell
def __(devices, np, problem_id, stoch_prob, y0):
    J_gpu = stoch_prob.subproblems[problem_id]

    tnt = J_gpu.layer.devices[0].time_horizon * (
        sum(d.num_devices * d.num_terminals_per_device for d in devices)
        + sum(
            d.num_devices * d.num_terminals_per_device for d in devices if d.is_ac
        )
    )

    J_gpu.layer.warm_start = False

    J_gpu.layer.solver.atol = 1.0e-3
    J_gpu.layer.solver.num_iterations = 1000
    J_gpu.layer.solver.rho_angle = 0.5
    J_gpu.layer.solver.alpha = 1.1
    J_gpu.layer.solver.scale_dual_residuals = False

    J_gpu.layer.solver.rho_power = 0.1 * y0.problem.value / np.sqrt(tnt)
    return J_gpu, tnt


@app.cell
def __(J_gpu):
    J_gpu.layer.solver.rho_power
    return


@app.cell
def __(J_gpu, theta0_gpu):
    s0 = J_gpu.layer(**theta0_gpu)
    return s0,


@app.cell
def __(J_gpu, np, s0, y0):
    s0  # Force dependency
    print("f_star:", y0.problem.value)
    print("f_admm:", J_gpu.layer.history.objective[-1])
    print(
        f"subopt: {100 * np.abs(J_gpu.layer.history.objective[-1] - y0.problem.value) / y0.problem.value:.03f} %"
    )
    return


@app.cell
def __():
    model_id = 1
    return model_id,


@app.cell
def __():
    problem_id = 0
    return problem_id,


@app.cell(hide_code=True)
def __(J_gpu, plot_convergence, s0, y0):
    s0  # Force dependency
    plot_convergence(J_gpu.layer.history, J_gpu.layer.solver, y0.problem.value)
    return


@app.cell
def __(J_gpu, zap):
    _battery = J_gpu.layer.devices[-2]

    zap.devices.store.C_matrix(_battery, 24, machine="cuda").shape
    return


if __name__ == "__main__":
    app.run()
