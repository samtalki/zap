import marimo

__generated_with = "0.7.1"
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

        total_primal = np.sqrt(np.power(hist.power, 2) + np.power(hist.phase, 2))
        total_dual = np.sqrt(
            np.power(hist.dual_power, 2) + np.power(hist.dual_phase, 2)
        )

        ax = axes[0][0]
        ax.hlines(
            admm.primal_tol,
            xmin=0,
            xmax=admm_num_iters,
            color="black",
            zorder=-100,
        )
        ax.hlines(
            admm.primal_tol_power,
            color="C0",
            xmin=0,
            xmax=admm_num_iters,
            zorder=-100,
        )
        ax.hlines(
            admm.primal_tol_angle,
            color="C1",
            xmin=0,
            xmax=admm_num_iters,
            zorder=-100,
        )

        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.plot(total_primal, color="black", ls="dashed")
        ax.set_yscale("log")
        ax.set_title("primal residuals")
        if ylims is not None:
            ax.set_ylim(*ylims)

        ax = axes[0][1]

        ax.hlines(
            admm.dual_tol, xmin=0, xmax=admm_num_iters, color="black", zorder=-100
        )
        ax.hlines(
            admm.dual_tol_power, xmin=0, xmax=admm_num_iters, color="C0", zorder=-100
        )
        ax.hlines(
            admm.dual_tol_angle, xmin=0, xmax=admm_num_iters, color="C1", zorder=-100
        )
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
    return plot_convergence,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Data")
    return


@app.cell
def __(zap):
    # net, devices = zap.importers.load_garver_network(line_slack=2.0, init_solar=50.0)

    # devices[2].nominal_capacity += 1.0
    # devices[2].susceptance /= np.median(devices[2].susceptance)

    # devices += [
    #     zap.Ground(
    #         num_nodes=net.num_nodes, terminal=np.array([0]), voltage=np.array([0.0])
    #     )
    # ]

    # for d in devices:
    #     d.scale_power(100.0)

    # ----

    # _config = runner.expand_config(
    #     runner.load_config("experiments/config/test_gpu_v09.yaml")
    # )[0]
    # # _config["data"]["use_batteries"] = False
    # _config["data"]["args"]["cost_unit"] = 100.0

    # _dataset = runner.load_dataset(**_config["data"])

    # net, devices = _dataset["net"], _dataset["devices"]

    # # devices[4].linear_cost += 0.005
    # # devices = [devices[i] for i in [0, 1, 4, 5]]
    # # devices[3].susceptance *= 100.0

    net, devices = zap.importers.load_battery_network()

    for d in devices:
        print(type(d))
    return d, devices, net


@app.cell
def __():
    common_params = {
        "emissions_weight": 200.0,
        "parameters": ["generator", "dc_line", "ac_line", "battery"],
    }
    return common_params,


@app.cell
def __(common_params, devices, net, runner):
    dataset_cvx = runner.setup_problem(
        net, devices, regularize=1e-6, **common_params
    )
    problem_cvx, layer_cvx = dataset_cvx["problem"], dataset_cvx["layer"]
    return dataset_cvx, layer_cvx, problem_cvx


@app.cell
def __():
    admm_args = {
        "machine": "cuda",
        "num_iterations": 5000,
        "minimum_iterations": 250,
        "atol": 1.0e-3,
        "verbose": False,
        "battery_window": 24,
        "alpha": 1.0,
        "relative_rho_angle": False,
        "rho_power": 1.0,  # / 100.0,
        "rho_angle": 1.0,  # 0.5 / 100.0,
        "adaptive_rho": True,
        "adaptation_tolerance": 2.0,
        "tau": 1.1,

        "battery_inner_iterations": 20,
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
        "algorithm": zap.planning.GradientDescent(step_size=1e-3 * 100.0, clip=1e3),
        "num_iterations": 25,
        "verbosity": 1,
    }
    return solver_kwargs,


@app.cell
def __():
    mosek_tol = 1e-8
    return mosek_tol,


@app.cell(hide_code=True)
def __(mosek_tol, problem_cvx, solver_kwargs):
    problem_cvx.layer.solver_kwargs["verbose"] = False
    problem_cvx.layer.solver_kwargs["mosek_params"][
        "MSK_DPAR_INTPNT_TOL_REL_GAP"
    ] = mosek_tol
    problem_cvx.layer.solver_kwargs["mosek_params"][
        "MSK_DPAR_INTPNT_TOL_DFEAS"
    ] = mosek_tol
    problem_cvx.layer.solver_kwargs["mosek_params"][
        "MSK_DPAR_INTPNT_TOL_PFEAS"
    ] = mosek_tol

    theta_cvx, history_cvx = problem_cvx.solve(
        num_iterations=solver_kwargs["num_iterations"],
        verbosity=1,
        algorithm=solver_kwargs["algorithm"],
        trackers=solver_kwargs["trackers"],
    )
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
    history_admm,
    history_cvx,
    problem_admm,
    problem_cvx,
):
    print("Loss CVX:\t\t", history_cvx["loss"][-1])
    print("Loss ADMM:\t\t", history_admm["loss"][-1], "\n")

    print("Invest CVX:\t\t", problem_cvx.inv_cost.item())
    print("Invest ADMM:\t", problem_admm.inv_cost.item(), "\n")

    print(
        "Emissions CVX:\t",
        common_params["emissions_weight"] * emissions_cvx.item(),
    )
    print(
        "Emissions ADMM:\t",
        common_params["emissions_weight"] * emissions_admm.item(),
        "\n",
    )

    print("Fuel CVX:\t\t", fuel_cvx.item())
    print("Fuel ADMM:\t\t", fuel_admm.item())
    return


@app.cell(hide_code=True)
def __(history_admm, history_cvx, np, plt):
    fig, ax = plt.subplots(figsize=(6, 2))

    ax.plot(history_cvx["loss"], label="cvx")
    ax.plot(history_admm["loss"], label="admm", ls="dashed")

    ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax.set_xticks(np.arange(25))

    # ax.set_yscale("log")
    # ax.set_ylim(10_000.0, 15_500.0)

    fig
    return ax, fig


@app.cell(hide_code=True)
def __(history_admm, history_cvx, problem_admm, problem_cvx, torch):
    history_admm, history_cvx  # Force dependency

    emissions_cvx = problem_cvx.operation_objective.objectives[1](
        problem_cvx.torch_state, parameters=problem_cvx.params, la=torch
    )
    emissions_admm = problem_admm.operation_objective.objectives[1](
        problem_admm.state, parameters=problem_admm.params, la=torch
    )
    return emissions_admm, emissions_cvx


@app.cell(hide_code=True)
def __(history_admm, history_cvx, problem_admm, problem_cvx, torch):
    history_admm, history_cvx  # Force dependency

    fuel_cvx = problem_cvx.operation_objective.objectives[0](
        problem_cvx.torch_state, parameters=problem_cvx.params, la=torch
    )
    fuel_admm = problem_admm.operation_objective.objectives[0](
        problem_admm.state, parameters=problem_admm.params, la=torch
    )
    return fuel_admm, fuel_cvx


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Analysis")
    return


@app.cell
def __():
    iteration = 3
    return iteration,


@app.cell
def __(emissions_admm, fuel_admm, history_admm, iteration):
    fuel_admm, emissions_admm  # Force dep

    theta_test = {
        k: v.detach().clone() + 1.0 for k, v in history_admm["param"][iteration].items()
    }

    # problem_admm.initialize_parameters(None)
    # zap.util.torchify(theta_cvx)
    # theta_admm
    return theta_test,


@app.cell
def __(layer_cvx, problem_cvx, theta_test):
    layer_mosek_tol = 1e-8

    layer_cvx.solver_kwargs["verbose"] = False
    layer_cvx.solver_kwargs["mosek_params"][
        "MSK_DPAR_INTPNT_TOL_REL_GAP"
    ] = layer_mosek_tol
    layer_cvx.solver_kwargs["mosek_params"][
        "MSK_DPAR_INTPNT_TOL_DFEAS"
    ] = layer_mosek_tol
    layer_cvx.solver_kwargs["mosek_params"][
        "MSK_DPAR_INTPNT_TOL_PFEAS"
    ] = layer_mosek_tol

    y_cvx = layer_cvx(
        **{k: v.detach().clone().cpu() for k, v in theta_test.items()}
    )
    f_cvx, grad_cvx = problem_cvx.forward_and_back(
        **{k: v.detach().clone().cpu().numpy() for k, v in theta_test.items()}
    )
    return f_cvx, grad_cvx, layer_mosek_tol, y_cvx


@app.cell
def __(full_cost, history_admm, iteration, layer_admm, theta_test):
    extra_unroll = False
    num_unroll_iterations = 250

    layer_admm.warm_start = False
    layer_admm.verbose = True

    layer_admm.solver.verbose = True
    layer_admm.solver.scale_dual_residuals = True
    layer_admm.solver.minimum_iterations = 250
    layer_admm.solver.num_iterations = 1000

    layer_admm.solver.atol = 1e-3
    layer_admm.solver.rtol = 0.0  # 1e-3
    layer_admm.solver.angle_converges_separately = False  # True

    layer_admm.solver.rho_power = 1.0  # problem_admm.rho_power_history[iteration + 1]
    layer_admm.solver.rho_angle = 1.0  # problem_admm.rho_angle_history[iteration + 1]
    layer_admm.solver.relative_rho_angle = False

    layer_admm.solver.adaptive_rho = True
    layer_admm.solver.adaptation_tolerance = 2.0
    layer_admm.solver.adaptation_frequency = 50
    layer_admm.solver.tau = 1.1

    layer_admm.solver.battery_window = 24
    layer_admm.solver.battery_inner_over_relaxation = 1.8
    layer_admm.solver.battery_inner_weight = 0.5
    layer_admm.solver.battery_inner_iterations = 10

    layer_admm.solver.alpha = 1.0

    # Gradient version of parameters
    theta_test2 = {k: v.clone().detach() for k, v in theta_test.items()}
    for k in theta_test2.keys():
        theta_test2[k].requires_grad = True

    _initial_state = history_admm["admm_state"][iteration - 1].copy()

    # Solve once to high accuracy
    if extra_unroll:
        y_admm0 = layer_admm(**theta_test, initial_state=_initial_state)

        # Solve again to unroll
        layer_admm.solver.adaptive_rho = False
        layer_admm.solver.minimum_iterations = num_unroll_iterations
        layer_admm.solver.num_iterations = num_unroll_iterations
        layer_admm.solver.atol = 1e-3
        y_admm = layer_admm(**theta_test2, initial_state=y_admm0)

    else:
        y_admm = layer_admm(**theta_test2, initial_state=_initial_state)

    # Differentiate
    f_admm = full_cost(y_admm, theta_test2)
    f_admm.backward()

    grad_admm = {k: v.grad for k, v in theta_test2.items()}
    return (
        extra_unroll,
        f_admm,
        grad_admm,
        k,
        num_unroll_iterations,
        theta_test2,
        y_admm,
        y_admm0,
    )


@app.cell
def __(layer_admm):
    layer_admm.devices[-2].rho
    return


@app.cell
def __(layer_admm, np, y_cvx):
    y_cvx.problem.value / np.sqrt(layer_admm.solver.num_dc_terminals)
    return


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
        layer_admm.history,
        layer_admm.solver,
        fstar=y_cvx.problem.value,
        ylims=None,
    )
    _fig
    return


@app.cell(hide_code=True)
def __(problem_admm, torch):
    def full_cost(y, theta):
        _p = problem_admm
        return _p.investment_objective(**theta, la=torch) + _p.operation_objective(
            y.as_outcome(), parameters=_p.layer.setup_parameters(**theta), la=torch
        )
    return full_cost,


@app.cell(hide_code=True)
def __(clips, np, plt, torch):
    def grad_plot(
        grad_cvx,
        grad_admm,
        iter=None,
        n1=0,
        n2=None,
        diff=False,
        title="Gradients",
        should_clip=False,
    ):
        num_plots = np.maximum(len(grad_cvx.keys()), 2)

        fig, axes = plt.subplots(num_plots, 1, figsize=(7, 1.5 * num_plots))

        names = ["cvx", "admm"]
        grads = [grad_cvx, grad_admm]

        for j, ky in enumerate(sorted(list(grad_cvx.keys()))):
            # Sort by grad cvx
            order = torch.argsort(grad_cvx[ky].ravel())

            if n2 is not None:
                order = order[n1:n2]

            x = np.arange(order.shape[0])

            if diff:
                axes[j].scatter(
                    x,
                    torch.clip(
                        grad_admm[ky].ravel().cpu()[order], -clips[ky], clips[ky]
                    )
                    - grad_cvx[ky].ravel()[order],
                    label="diff",
                    s=5,
                )

            else:
                axes[j].plot(x, grad_cvx[ky].ravel()[order], label="cvx")
                if should_clip:
                    dx = torch.clip(
                        grad_admm[ky].ravel().cpu()[order], -clips[ky], clips[ky]
                    )
                else:
                    dx = grad_admm[ky].ravel().cpu()[order]

                axes[j].scatter(
                    x,
                    dx,
                    label="admm",
                    c="red",
                    s=5,
                )

            axes[j].set_ylabel(ky, fontsize=10)
            if n2 is not None:
                axes[j].set_xlim(n1, n2)

        axes[0].legend(fontsize=10)
        # axes[0].set_ylim(-5, 5)

        if iter is None:
            fig.suptitle(f"{title}")
        else:
            fig.suptitle(f"{title} on iteration {iter}", fontsize=10)
        fig.tight_layout()

        return fig, axes
    return grad_plot,


@app.cell(hide_code=True)
def __(clips, grad_admm, grad_cvx, layer_admm, torch):
    _subnorms = [
        torch.linalg.vector_norm(
            torch.clip(grad_admm[k].cpu(), -clips[k], clips[k]) - grad_cvx[k], 2
        )
        for k in grad_admm.keys()
    ]
    _err = torch.linalg.vector_norm(torch.tensor(_subnorms)).item()

    _subnorms = [
        torch.linalg.vector_norm(grad_cvx[k], 2) for k in grad_admm.keys()
    ]
    _total_grad = torch.linalg.vector_norm(torch.tensor(_subnorms)).item()

    print(f"Tolerance:\t\t\t {layer_admm.solver.atol:.1e} / {layer_admm.solver.rtol:.1e}")
    print("Gradient Error:\t\t", _err)
    print("Total Gradient:\t\t", _total_grad)
    return


@app.cell(hide_code=True)
def __(layer_admm, np, y_admm, y_cvx):
    _nu1 = y_admm.dual_power.detach().cpu() * layer_admm.solver.rho_power
    _nu2 = y_cvx.prices

    print("Price Error:\t", np.linalg.norm((_nu1 + _nu2).ravel()))
    print("Price Norm:\t\t", np.linalg.norm(_nu2.ravel()))
    return


@app.cell
def __(grad_admm, torch):
    clips = {k: 10.0 * torch.median(torch.abs(v)).item() for k, v in grad_admm.items()}
    return clips,


@app.cell
def __(devices):
    z = devices[0]
    return z,


@app.cell(hide_code=True)
def __(devices, grad_plot, iteration, torch, y_admm, y_cvx):
    _iter = iteration
    _fig, _axes = grad_plot(
        {
            f"{type(devices[i]).__name__}": torch.Tensor(y_cvx.power[i][0])
            for i in range(len(devices))
        },
        {
            f"{type(devices[i]).__name__}": y_admm.power[i][0].detach()
            for i in range(len(devices))
        },
        iter=_iter,
        diff=False,
        title="Variable"
    )
    _fig
    return


@app.cell
def __(grad_admm, grad_cvx, grad_plot, iteration):
    _iter = iteration
    _fig, _axes = grad_plot(
        grad_cvx,
        grad_admm,
        iter=_iter,
        diff=False,
        should_clip=True,
    )

    # for _ax in _axes:
    #     _ax.set_ylim(*_axes[-1].get_ylim())

    _fig
    return


@app.cell(hide_code=True)
def __(layer_admm, np, y_admm, y_cvx):
    _op = np.linalg.norm

    primal_scale_power = _op([_op([_op(pi) for pi in p]) for p in y_cvx.power])
    primal_scale_angle = _op(
        [_op([_op(pi) for pi in p]) for p in y_cvx.angle if p is not None]
    )

    dual_scale_power = layer_admm.solver.rho_power * _op(
        (y_admm.num_terminals * y_admm.dual_power).detach().cpu()
    )
    dual_scale_angle = layer_admm.solver.rho_angle * _op(
        [_op([_op(pi) for pi in p]) for p in y_cvx.phase_duals if p is not None]
    )

    print("Power:", primal_scale_power, "-", dual_scale_power)
    print("Angle:", primal_scale_angle, "-", dual_scale_angle)
    return (
        dual_scale_angle,
        dual_scale_power,
        primal_scale_angle,
        primal_scale_power,
    )


@app.cell(hide_code=True)
def __():
    # _rhop = 1.0
    # _rhov = 1.0
    # _op = np.inf

    # x1, v1 = problem_cvx.layer.devices[3].cvx_admm_prox_update(
    #     _rhop,
    #     _rhov,
    #     [p.detach().cpu() for p in y_admm.power[3]],
    #     [p.detach().cpu() for p in y_admm.phase[3]],
    #     theta_test["ac_line"].cpu(),
    # )

    # problem_admm.layer.devices[3].has_changed = True
    # x2, v2 = problem_admm.layer.devices[3].admm_prox_update(
    #     _rhop,
    #     _rhov,
    #     y_admm.power[3],
    #     y_admm.phase[3],
    #     theta_test["ac_line"]
    # )

    # print(np.linalg.norm((x1[0] - x2[0].detach().cpu().numpy()).ravel(), _op))
    # print(np.linalg.norm((x1[1] - x2[1].detach().cpu().numpy()).ravel(), _op))

    # print(np.linalg.norm((v1[0] - v2[0].detach().cpu().numpy()).ravel(), _op))
    # print(np.linalg.norm((v1[1] - v2[1].detach().cpu().numpy()).ravel(), _op))

    # inds = np.argsort(-np.abs((x1[0] - x2[0].detach().cpu().numpy()) ), axis=0)[:10, 4]
    return


@app.cell
def __():
    # layer_admm.solver.primal_tol_power, layer_admm.solver.dual_tol_power
    # layer_admm.solver.primal_tol_angle, layer_admm.solver.dual_tol_angle
    return


@app.cell
def __():
    # devices[0].nominal_capacity, theta_admm["generator"], theta_cvx["generator"]
    return


@app.cell
def __():
    # history_admm  # force dep
    # problem_admm.state.power[1]
    return


if __name__ == "__main__":
    app.run()
