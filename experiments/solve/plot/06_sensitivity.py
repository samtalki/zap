import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import pandas as pd
    import scipy.sparse as sp

    import torch
    import importlib
    import pypsa
    import datetime as dt

    from copy import deepcopy
    from pathlib import Path
    return Path, cp, deepcopy, dt, importlib, mo, np, pd, pypsa, sp, torch


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


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"## Data")
    return


@app.cell
def __():
    num_days = 1
    return num_days,


@app.cell
def __(pypsa):
    pn = pypsa.Network()
    pn.import_from_csv_folder("data/pypsa/western/load_medium/elec_s_500")
    # pn.storage_units = pn.storage_units[pn.storage_units.p_nom > 0.0]
    return pn,


@app.cell
def __(dt, pd, zap):
    def load_pypsa_network(
        pn,
        time_horizon=1,
        start_date=dt.datetime(2019, 1, 2, 0),
        exclude_batteries=False,
        **pypsa_kwargs,
    ):
        dates = pd.date_range(
            start_date,
            start_date + dt.timedelta(hours=time_horizon),
            freq="1h",
            inclusive="left",
        )

        net, devices = zap.importers.load_pypsa_network(pn, dates, **pypsa_kwargs)
        if exclude_batteries:
            devices = devices[:-1]

        return net, devices, time_horizon
    return load_pypsa_network,


@app.cell
def __(dt, load_pypsa_network, np, num_days, pn, zap):
    net, devices, time_horizon = load_pypsa_network(
        pn,
        time_horizon=24 * num_days,
        start_date=dt.datetime(2019, 8, 9, 7),  # dt.datetime(2019, 8, 9, 7),
        exclude_batteries=False,
        # Units
        power_unit=1000.0,
        cost_unit=100.0,
        # Costs
        marginal_load_value=500.0,
        load_cost_perturbation=10.0,
        generator_cost_perturbation=1.0,
        # Rescale capacities
        scale_load=0.5,
        scale_generator_capacity_factor=0.7,
        scale_line_capacity_factor=0.7,
        # Empty generators
        drop_empty_generators=False,
        expand_empty_generators=0.5,
        # Battery stuff
        battery_discharge_cost=1.0,
        battery_init_soc=0.0,
        battery_final_soc=0.0,
    )
    _ground = zap.Ground(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        voltage=np.array([0.0]),
    )
    devices += [_ground]

    for d in devices:
        if hasattr(d, "capital_cost") and d.capital_cost is not None:
            d.capital_cost *= 0.0
    return d, devices, net, time_horizon


@app.cell
def __(torch):
    machine, dtype = "cuda", torch.float32
    return dtype, machine


@app.cell
def __(devices, dtype, machine, torch):
    torch.cuda.empty_cache()

    torch_devices = [d.torchify(machine=machine, dtype=dtype) for d in devices]
    return torch_devices,


@app.cell
def __(devices, np):
    total_load = -np.sum(devices[1].min_power * devices[1].nominal_capacity)
    total_load
    return total_load,


@app.cell(hide_code=True)
def __(dt, load_pypsa_network, np, pn):
    _, _devices, _T = load_pypsa_network(
        pn,
        time_horizon=24 * 360,
        start_date=dt.datetime(2019, 1, 1, 7),
        # Units
        power_unit=1000.0,
        cost_unit=100.0,
        # Costs
        marginal_load_value=500.0,
        load_cost_perturbation=10.0,
        generator_cost_perturbation=1.0,
        # Rescale capacities
        scale_load=0.6,
        scale_generator_capacity_factor=0.7,
        scale_line_capacity_factor=0.7,
        # Empty generators
        drop_empty_generators=False,
        expand_empty_generators=0.5,
        # Battery stuff
        battery_discharge_cost=1.0,
        battery_init_soc=0.0,
        battery_final_soc=0.0,
    )

    _total = -np.sum(_devices[1].min_power * _devices[1].nominal_capacity)
    _hourly = _total / _T

    print(f"Average Hourly Load: {_hourly} GW")
    return


@app.cell
def __(mo):
    mo.md("""## Build Layer""")
    return


@app.cell
def __():
    from zap.planning import PlanningProblem, DispatchCostObjective, InvestmentObjective
    return DispatchCostObjective, InvestmentObjective, PlanningProblem


@app.cell
def __():
    from zap.admm import ADMMSolver, ADMMLayer
    return ADMMLayer, ADMMSolver


@app.cell
def __(ADMMSolver, torch):
    admm = ADMMSolver(
        machine="cuda",
        dtype=torch.float32,
        resid_norm=2,
        battery_window=24,
        num_iterations=100,
        atol=1e-8,
        rho_power=1.0,
        rho_angle=1.0,
        alpha=1.0,
        adaptive_rho=True,
    )
    return admm,


@app.cell
def __(
    ADMMLayer,
    DispatchCostObjective,
    InvestmentObjective,
    PlanningProblem,
    admm,
    deepcopy,
    net,
    torch_devices,
):
    parameters = {
        "generator_capacity": (0, "nominal_capacity"),
        "ac_line_capacity": (3, "nominal_capacity"),
    }

    layer = ADMMLayer(
        net,
        torch_devices,
        parameters,
        solver=deepcopy(admm),
        time_horizon=24,
        warm_start=False,
    )

    problem = PlanningProblem(
        DispatchCostObjective(net, torch_devices),
        InvestmentObjective(torch_devices, layer),
        layer,
    )
    return layer, parameters, problem


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Parameters""")
    return


@app.cell
def __(backprop, parameters, torch_devices):
    param0_nograd = {
        k: getattr(torch_devices[i], name).clone().detach()
        for k, (i, name) in parameters.items()
    }

    param0 = {
        k: getattr(torch_devices[i], name).clone().detach()
        for k, (i, name) in parameters.items()
    }

    if backprop:
        for k, p in param0.items():
            p.requires_grad = True
    return k, p, param0, param0_nograd


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Compute True Gradients via Mosek""")
    return


@app.cell
def __(
    DispatchCostObjective,
    InvestmentObjective,
    PlanningProblem,
    cp,
    devices,
    net,
    parameters,
    regularize,
    zap,
):
    layer_cvx = zap.DispatchLayer(
        net,
        devices,
        parameters,
        time_horizon=24,
        solver=cp.MOSEK,
    )

    problem_cvx = PlanningProblem(
        DispatchCostObjective(net, devices),
        InvestmentObjective(devices, layer_cvx),
        layer_cvx,
        regularize=regularize,
    )
    return layer_cvx, problem_cvx


@app.cell
def __(param0, problem_cvx):
    y_cvx, grad_cvx = problem_cvx.forward_and_back(**{k: v.detach().cpu().numpy() for k,v in param0.items()})
    return grad_cvx, y_cvx


@app.cell
def __(y_cvx):
    print(y_cvx)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Approximate Gradients with Unrolling""")
    return


@app.cell
def __():
    backprop = True
    return backprop,


@app.cell
def __(backprop, devices, num_days):
    print(
        f"Solving for {num_days*24} hours and {sum(d.num_devices for d in devices)} devices."
    )
    if backprop:
        print("Gradient tape enabled.")
    else:
        print("Not tracking gradients.")
    return


@app.cell(hide_code=True)
def __(
    ADMMLayer,
    DispatchCostObjective,
    InvestmentObjective,
    PlanningProblem,
    admm,
    deepcopy,
    net,
    parameters,
    torch,
    torch_devices,
    total_load,
):
    def get_solution_and_grad(param, num_iterations, verbose=True, no_grad=False):
        layer = ADMMLayer(
            net,
            torch_devices,
            parameters,
            solver=deepcopy(admm),
            time_horizon=24,
            warm_start=False,
            verbose=verbose,
        )
        layer.solver.verbose = verbose

        problem = PlanningProblem(
            DispatchCostObjective(net, torch_devices),
            InvestmentObjective(torch_devices, layer),
            layer,
        )
        problem.layer.solver.num_iterations = num_iterations

        if no_grad:
            y0 = problem.forward(**param)
            grad0 = None
        else:
            y0, grad0 = problem.forward_and_back(**param)

        print(f"\n\nTotal Load: {total_load} GW")
        print(
            f"Total Abs Nodal Power Imbalance: {torch.sum(torch.abs(layer.state.avg_power))} GW"
        )

        return y0, grad0
    return get_solution_and_grad,


@app.cell
def __():
    num_iter_list = [10, 100, 1000]
    return num_iter_list,


@app.cell
def __(get_solution_and_grad, param0):
    _ = get_solution_and_grad(param0, 1000, verbose=False)
    return


@app.cell
def __(get_solution_and_grad, param0_nograd):
    _ = get_solution_and_grad(param0_nograd, 1000, verbose=False, no_grad=True)
    return


@app.cell
def __(get_solution_and_grad, num_iter_list, param0):
    y_list, grad_list = zip(*[get_solution_and_grad(param0, i) for i in num_iter_list])
    return grad_list, y_list


@app.cell
def __():
    # _y, _grad = get_solution_and_grad(param0, 100)
    return


@app.cell
def __():
    # y0, grad0 = problem.forward_and_back(**param0)

    # print(f"\n\nTotal Load: {total_load} GW")
    # print(
    #     f"Total Abs Nodal Power Imbalance: {torch.sum(torch.abs(layer.state.avg_power))} GW"
    # )

    # plot_convergence(layer.history, eps_pd=layer.solver.primal_tol, fstar=None)
    return


@app.cell(hide_code=True)
def __(np, plt):
    def plot_convergence(hist, eps_pd=None, fstar=None):
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

        print(f"Primal Resid:\t\t {hist.power[-1] + hist.phase[-1]}")
        print(f"Dual Resid:\t\t\t {hist.dual_power[-1] + hist.dual_phase[-1]}")
        print(f"Objective:\t\t\t {hist.objective[-1]}")

        admm_num_iters = len(hist.power)

        ax = axes[0]
        if eps_pd is not None:
            ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title("primal residuals")

        ax = axes[1]
        if eps_pd is not None:
            ax.hlines(eps_pd, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.dual_power, label="power")
        ax.plot(hist.dual_phase, label="angle")
        ax.set_yscale("log")
        # ax.legend(fontsize=8)
        ax.set_title("dual residuals")

        ax = axes[2]
        ax.plot(np.array(hist.objective))
        ax.set_title("f")
        if fstar is not None:
            ax.hlines(fstar, xmin=0, xmax=len(hist.objective), color="black")
            print(f"Optimal Objective:\t {fstar}")
            print(f"Objective Gap:\t\t {np.abs(hist.objective[-1] - fstar) / np.abs(fstar)}")

        # ax = axes[1][1]
        # if len(hist.price_error) > 0:
        #     ax.plot(np.array(hist.price_error) / simple_result.prices.size)
        # ax.set_yscale("log")
        # ax.set_title("nu - nu*")

        fig.tight_layout()
        return fig
    return plot_convergence,


@app.cell
def __():
    regularize = 1e-8
    return regularize,


@app.cell
def __(
    Path,
    devices,
    grad_cvx,
    grad_list,
    np,
    num_iter_list,
    plt,
    problem_cvx,
):
    def grad_plot(n=50, key="generator_capacity"):
        fig, ax = plt.subplots(figsize=(6.5, 3))

        # Order gradients by CVX
        dy = grad_cvx[key].numpy().ravel()[:n]
        order = np.argsort(dy)

        # Verify against duals
        lam = -np.sum(problem_cvx.state.local_inequality_duals[0][1] * devices[0].max_power, axis=1)
        lam += devices[0].capital_cost.ravel()

        # Plot CVX gradient
        ax.bar(range(n), dy[order], label="True Gradient")
        print("Dual minus implicit grad:", np.linalg.norm(lam[:n] - dy) / np.linalg.norm(lam[:n]))

        # Plot ADMM gradient estimates
        for i in range(len(num_iter_list)):
            ax.scatter(
                range(n),
                grad_list[i][key].cpu().numpy().ravel()[:n][order],
                s=8,
                label=f"{num_iter_list[i]} Iterations",
            )

        ax.legend(loc="upper center", framealpha=1)
        ax.set_title("Unrolled Gradients of Generator Capacity on Total Cost")
        ax.set_xlabel("Generator Index")

        fig.tight_layout()
        fig.savefig(Path().home() / "figures/gpu/sensitivity.pdf")

        return fig


    grad_plot()
    return grad_plot,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Debug""")
    return


@app.cell
def __(pypsa):
    pn2 = pypsa.Network()
    pn2.import_from_csv_folder("data/pypsa/western/load_medium/elec_s_4000")
    return pn2,


@app.cell
def __(load_pypsa_network, pn2):
    _net, devs2, _time = load_pypsa_network(pn2)

    print(f"This case has {sum(d.num_devices for d in devs2)} devices.")
    return devs2,


@app.cell
def __(devs2):
    devs2[3].num_devices
    return


@app.cell
def __():
    # param1 = deepcopy(param0)

    # for _p in param1:
    #     for k, v in _p.items():
    #         _p[k] = v.clone().detach() + torch.rand(v.shape, device="cuda") * 0.010
    return


@app.cell
def __():
    # state1, history1 = admm.solve(
    #     net,
    #     torch_devices,
    #     time_horizon,
    #     parameters=param1,
    #     initial_state=state.copy(),
    # )
    return


@app.cell
def __():
    # plot_convergence(history1, eps_pd=admm.rtol * np.sqrt(admm.total_terminals))
    return


if __name__ == "__main__":
    app.run()
