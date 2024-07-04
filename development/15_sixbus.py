import marimo

__generated_with = "0.6.25"
app = marimo.App(app_title="Six Bus System")


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
    import matplotlib.pyplot as plt
    import seaborn
    seaborn.set_theme(style="whitegrid")
    return plt, seaborn


@app.cell
def __():
    import zap
    return zap,


@app.cell
def __(importlib):
    from experiments import runner
    _ = importlib.reload(runner)
    return runner,


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
    mo.md("## Data")
    return


@app.cell
def __(np, zap):
    net, devices = zap.importers.load_garver_network(line_slack=2.0, init_solar=50.0)

    devices[2].nominal_capacity += 1.0
    devices[2].susceptance /= np.median(devices[2].susceptance)

    devices += [
        zap.Ground(
            num_nodes=net.num_nodes, terminal=np.array([0]), voltage=np.array([0.0])
        )
    ]

    for d in devices:
        d.scale_power(100.0)
    return d, devices, net


@app.cell
def __():
    common_params = {
        "emissions_weight": 10.0,
        "parameters": ["generator", "ac_line"],
    }
    return common_params,


@app.cell
def __(common_params, devices, net, runner):
    dataset_cvx = runner.setup_problem(net, devices, regularize=0.0, **common_params)
    problem_cvx, layer_cvx = dataset_cvx["problem"], dataset_cvx["layer"]
    return dataset_cvx, layer_cvx, problem_cvx


@app.cell
def __():
    admm_args = {
        "machine": "cpu",
        "num_iterations": 5000,
        "rho_power": 1.0,
        "adaptive_rho": True,
        "adaptation_tolerance": 2.0,
        "tau": 1.1,
        "alpha": 1.0,
        "atol": 1.0e-6,
        "verbose": False,
        "relative_rho_angle": True,
        "rho_angle": 0.5,
        "minimum_iterations": 100,
    }
    return admm_args,


@app.cell
def __(admm_args, common_params, devices, net, runner):
    dataset_admm = runner.setup_problem(
        net, devices, use_admm=True, args=admm_args, **common_params
    )
    problem_admm, layer_admm = dataset_admm["problem"], dataset_admm["layer"]
    return dataset_admm, layer_admm, problem_admm


@app.cell
def __(mo):
    mo.md("## Solver")
    return


@app.cell
def __():
    import zap.planning.trackers as tr
    return tr,


@app.cell
def __(tr, zap):
    solver_kwargs = {
        "trackers": tr.DEFAULT_TRACKERS + [tr.GRAD, tr.PARAM, tr.ADMM_STATE],
        "algorithm": zap.planning.GradientDescent(step_size=1e-3, clip=1e3),
        "num_iterations": 200,
        "verbosity": 1
    }
    return solver_kwargs,


@app.cell
def __(problem_cvx, solver_kwargs):
    theta_cvx, history_cvx = problem_cvx.solve(**solver_kwargs)
    return history_cvx, theta_cvx


@app.cell
def __(problem_admm, solver_kwargs):
    theta_admm, history_admm = problem_admm.solve(**solver_kwargs)
    return history_admm, theta_admm


@app.cell(hide_code=True)
def __(
    common_params,
    emissions_admm,
    emissions_cvx,
    fuel_admm,
    fuel_cvx,
    problem_admm,
    problem_cvx,
):
    print("Loss CVX:\t\t", problem_cvx.cost.item())
    print("Loss ADMM:\t\t", problem_admm.cost.item(), "\n")

    print("Invest CVX:\t\t", problem_cvx.inv_cost.item())
    print("Invest ADMM:\t", problem_admm.inv_cost.item(), "\n")

    print("Emissions CVX:\t", common_params["emissions_weight"] * emissions_cvx.item())
    print("Emissions ADMM:\t", common_params["emissions_weight"] * emissions_admm.item(), "\n")

    print("Fuel CVX:\t\t", fuel_cvx.item())
    print("Fuel ADMM:\t\t", fuel_admm.item())
    return


@app.cell(hide_code=True)
def __(history_admm, history_cvx, plt):
    fig, ax = plt.subplots(figsize=(6, 2))

    ax.plot(history_cvx["loss"], label="cvx")
    ax.plot(history_admm["loss"], label="admm", ls="dashed")
    # ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylim(10.0, 250.0)

    fig
    return ax, fig


@app.cell(hide_code=True)
def __(history_admm, problem_admm, problem_cvx, torch):
    history_admm  # Force dependency

    emissions_cvx = problem_cvx.operation_objective.objectives[1](
        problem_cvx.torch_state, parameters=problem_cvx.params, la=torch
    )
    emissions_admm = problem_cvx.operation_objective.objectives[1](
        problem_admm.state, parameters=problem_admm.params, la=torch
    )
    return emissions_admm, emissions_cvx


@app.cell(hide_code=True)
def __(history_admm, problem_admm, problem_cvx, torch):
    history_admm  # Force dependency

    fuel_cvx = problem_cvx.operation_objective.objectives[0](
        problem_cvx.torch_state, parameters=problem_cvx.params, la=torch
    )
    fuel_admm = problem_cvx.operation_objective.objectives[0](
        problem_admm.state, parameters=problem_admm.params, la=torch
    )
    return fuel_admm, fuel_cvx


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Analysis")
    return


@app.cell
def __():
    iteration = 8
    return iteration,


@app.cell
def __(history_admm, iteration):
    theta_test = {k: v.detach() for k, v in history_admm["param"][iteration].items()}

    # problem_admm.initialize_parameters(None)
    # zap.util.torchify(theta_cvx)
    # theta_admm
    return theta_test,


@app.cell
def __(layer_cvx, problem_cvx, theta_test):
    y_cvx = layer_cvx(**{k: v.detach().clone() for k, v in theta_test.items()})
    f_cvx, grad_cvx = problem_cvx.forward_and_back(
        **{k: v.detach().clone().numpy() for k, v in theta_test.items()}
    )
    return f_cvx, grad_cvx, y_cvx


@app.cell
def __(
    full_cost,
    history_admm,
    iteration,
    layer_admm,
    problem_admm,
    theta_test,
):
    layer_admm.warm_start = False
    layer_admm.verbose = True

    layer_admm.solver.verbose = False
    layer_admm.solver.rho_power = problem_admm.rho_history[iteration + 1]
    layer_admm.solver.minimum_iterations = 100
    layer_admm.solver.atol = 1e-6

    theta_test2 = {k: v.clone().detach() for k,v in theta_test.items()}
    for k in theta_test2.keys():
        theta_test2[k].requires_grad = True

    _initial_state = history_admm["admm_state"][iteration - 1].copy()

    y_admm = layer_admm(**theta_test2, initial_state=_initial_state)
    f_admm = full_cost(y_admm, theta_test2)
    f_admm.backward()

    grad_admm = {k: v.grad for k, v in theta_test2.items()}
    return f_admm, grad_admm, k, theta_test2, y_admm


@app.cell(hide_code=True)
def __(problem_admm, torch):
    def full_cost(y, theta):
        _p = problem_admm
        return _p.investment_objective(**theta, la=torch) + _p.operation_objective(
            y.as_outcome(), parameters=_p.layer.setup_parameters(**theta), la=torch
        )
    return full_cost,


@app.cell(hide_code=True)
def __(
    full_cost,
    history_admm,
    iteration,
    layer_admm,
    theta_test,
    y_admm,
    y_cvx,
):
    print("History:\t", full_cost(y_admm, theta_test).item())
    print("Reproduced:\t", history_admm["loss"][iteration][0])
    print()

    print("CVX Cost:\t", y_cvx.problem.value)
    print("ADMM Cost:\t", layer_admm.history.objective[-1])
    return


@app.cell(hide_code=True)
def __(layer_admm, plot_convergence, y_admm, y_cvx):
    y_admm  # Force dependency
    _fig = plot_convergence(
        layer_admm.history, layer_admm.solver, fstar=y_cvx.problem.value, ylims=(1e-4, 1e2)
    )
    _fig
    return


@app.cell(hide_code=True)
def __(np, plt):
    def grad_plot(grad_cvx, grad_admm, iter=None):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), width_ratios=[3, 1])

        names = ["cvx", "admm"]
        grads = [grad_cvx, grad_admm]

        for i in range(len(grads)):
            grad = grads[i]

            for j, ky in enumerate(sorted(list(grad.keys()))):
                axes[j].bar(
                    np.arange(grad[ky].shape[0]) + 0.35 * i,
                    grad[ky].ravel(),
                    label=names[i],
                    width=0.35,
                )
                axes[j].set_xticklabels([])
                axes[j].set_xlabel(ky, fontsize=10)

        axes[0].legend(fontsize=10)
        # axes[0].set_ylim(-5, 5)

        if iter is None:
            fig.suptitle("Gradients")
        else:
            fig.suptitle(f"Gradients on iteration {iter}", fontsize=10)
        fig.tight_layout()

        return fig
    return grad_plot,


@app.cell(hide_code=True)
def __(grad_admm, grad_cvx, layer_admm, torch):
    _subnorms = [
        torch.linalg.vector_norm(grad_admm[k] - grad_cvx[k], 2) ** 2
        for k in grad_admm.keys()
    ]
    _err = torch.linalg.vector_norm(torch.tensor(_subnorms)).item()

    print(f"Tolerance:\t\t\t {layer_admm.solver.atol:.1e}")
    print("Gradient Error:\t\t", _err)
    return


@app.cell
def __(grad_admm, grad_cvx, grad_plot, iteration):
    _iter = iteration
    grad_plot(
        grad_cvx, grad_admm, iter=_iter
    )
    return


@app.cell
def __(devices, theta_admm, theta_cvx):
    devices[0].nominal_capacity, theta_admm["generator"], theta_cvx["generator"]
    return


@app.cell
def __(history_admm, problem_admm):
    history_admm  # force dep
    problem_admm.state.power[1]
    return


if __name__ == "__main__":
    app.run()
