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
    def plot_convergence(hist, admm, fstar=1.0, ylims=(1e-3, 1e0)):
        fig, axes = plt.subplots(2, 2, figsize=(7, 3.5))

        admm_num_iters = len(hist.objective)
        eps_pd = admm.total_tol  # * np.sqrt(admm.total_terminals)

        total_primal = np.sqrt(np.power(hist.power, 2) + np.power(hist.phase, 2))
        total_dual = np.sqrt(
            np.power(hist.dual_power, 2) + np.power(hist.dual_phase, 2)
        )

        ax = axes[0][0]
        ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.plot(total_primal, color="black", ls="dashed")
        ax.set_yscale("log")
        ax.set_title("primal residuals")
        ax.set_ylim(*ylims)

        ax = axes[0][1]

        ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.dual_power, label="power")
        ax.plot(hist.dual_phase, label="angle")
        ax.plot(total_dual, color="black", ls="dashed")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("dual residuals")
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
def __():
    force_dc = False
    return force_dc,


@app.cell
def __(zap):
    dev_types = [zap.Generator, zap.Load, zap.Battery, zap.Ground, zap.DCLine, zap.ACLine]
    return dev_types,


@app.cell
def __(config, dev_types, force_dc, runner, zap):
    data = runner.load_dataset(**config["data"])
    net, devices = data["net"], data["devices"]

    devices = [d for d in devices if type(d) in dev_types]


    if force_dc:
        for i, _d in enumerate(devices):
            if isinstance(_d, zap.ACLine):
                devices[i] = zap.DCLine(
                    num_nodes=_d.num_nodes,
                    source_terminal=_d.source_terminal,
                    sink_terminal=_d.sink_terminal,
                    capacity=_d.capacity,
                    nominal_capacity=_d.nominal_capacity,
                    capital_cost=_d.capital_cost,
                    slack=_d.slack,
                    min_nominal_capacity=_d.min_nominal_capacity,
                    max_nominal_capacity=_d.max_nominal_capacity
                )
    return data, devices, i, net


@app.cell
def __(devices):
    for d in devices:
        print(type(d))
    return d,


@app.cell
def __(config, devices, net, runner):
    cpu_problem_data = runner.setup_problem(net, devices, **config["problem"])
    cpu_full_prob, cpu_stoch_prob = (
        cpu_problem_data["problem"],
        cpu_problem_data["stochastic_problem"],
    )
    return cpu_full_prob, cpu_problem_data, cpu_stoch_prob


@app.cell
def __(config, devices, net, runner, torch):
    problem_data = runner.setup_problem(
        net, devices, **config["problem"], **config["layer"]
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
def __(devices, np, problem_id, stoch_prob, theta0_gpu, y0):
    J_gpu = stoch_prob.subproblems[problem_id]

    tnt = J_gpu.layer.devices[0].time_horizon * (
        sum(d.num_devices * d.num_terminals_per_device for d in devices)
        + sum(
            d.num_devices * d.num_terminals_per_device for d in devices if d.is_ac
        )
    )

    J_gpu.layer.warm_start = False

    J_gpu.layer.solver.atol = 1.0e-4
    J_gpu.layer.solver.num_iterations = 1000
    J_gpu.layer.solver.relative_rho_angle = True
    J_gpu.layer.solver.rho_angle = 0.25
    J_gpu.layer.solver.alpha = 1.5
    J_gpu.layer.solver.scale_dual_residuals = True

    J_gpu.layer.solver.rho_power = 0.1 * y0.problem.value / np.sqrt(tnt)
    J_gpu.layer.solver.adaptive_rho = True
    J_gpu.layer.solver.adaptation_tolerance = 2.0
    J_gpu.layer.solver.tau = 1.1
    J_gpu.layer.solver.verbose = False

    s0 = J_gpu.layer(**theta0_gpu)
    J_gpu.layer.solver.rho_power
    return J_gpu, s0, tnt


@app.cell(hide_code=True)
def __(J_gpu, np, s0, y0):
    s0  # Force dependency
    print("f_star:", y0.problem.value)
    print("f_admm:", J_gpu.layer.history.objective[-1])
    print(
        f"subopt: {100 * np.abs(J_gpu.layer.history.objective[-1] - y0.problem.value) / y0.problem.value:.03f} %"
    )
    return


@app.cell
def __(J_gpu, plot_convergence, s0, y0):
    s0  # Force dependency
    plot_convergence(J_gpu.layer.history, J_gpu.layer.solver, y0.problem.value)
    return


@app.cell
def __():
    model_id = 300  # 1, 30, 300
    return model_id,


@app.cell
def __():
    problem_id = 0  # 0, 1, 8, 9, 10, 11, 26, 32
    return problem_id,


if __name__ == "__main__":
    app.run()
