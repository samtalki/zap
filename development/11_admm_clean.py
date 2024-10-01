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
    return cp, deepcopy, dt, importlib, mo, np, pd, pypsa, sp, torch


@app.cell
def __():
    import zap
    return zap,


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_theme()
    return plt, seaborn


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"## Data")
    return


@app.cell
def __():
    num_days = 8
    return num_days,


@app.cell
def __(pypsa):
    pn = pypsa.Network()
    pn.import_from_csv_folder("data/pypsa/western/load_medium/elec_s_4000")
    # pn.storage_units = pn.storage_units[pn.storage_units.p_nom > 0.0]
    return pn,


@app.cell(hide_code=True)
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
        start_date=dt.datetime(2019, 8, 9, 7),
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
    _ground = zap.Ground(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        voltage=np.array([0.0]),
    )
    devices += [_ground]
    return devices, net, time_horizon


@app.cell
def __(torch):
    machine, dtype = "cuda", torch.float32
    return dtype, machine


@app.cell
def __(devices, dtype, machine, torch):
    torch.cuda.empty_cache()

    torch_devices = [d.torchify(machine=machine, dtype=dtype) for d in devices]
    return torch_devices,


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Parameters""")
    return


@app.cell
def __(backprop, devices, torch_devices, zap):
    param0 = [{} for d in devices]

    for i in [0, 2, 3, 4]:
        if i == 4:
            assert isinstance(devices[i], zap.Battery)
            p = torch_devices[i].power_capacity.clone().detach()
        else:    
            p = torch_devices[i].nominal_capacity.clone().detach()

        if backprop:
            p.requires_grad = True

        if i == 4:
            param0[i]["power_capacity"] = p # + 0.001
        else:    
            param0[i]["nominal_capacity"] = p
    return i, p, param0


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Baseline Solve""")
    return


@app.cell
def __(cp, devices, net, param0):
    y_cvx = net.dispatch(
        devices,
        add_ground=False,
        solver=cp.MOSEK,
        parameters=[{k: v.cpu() for k, v in p.items()} for p in param0],
    )
    return y_cvx,


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Solve Part 1""")
    return


@app.cell
def __():
    from zap.admm import ADMMSolver
    return ADMMSolver,


@app.cell
def __(backprop, torch_devices):
    if backprop:
        for d in torch_devices:
            if hasattr(d, "nominal_capacity"):
                d.nominal_capacity.requires_grad = True
    return d,


@app.cell
def __():
    backprop = False
    return backprop,


@app.cell
def __(ADMMSolver, backprop, param0, torch):
    if backprop:
        print("Gradient tape enabled.")

    param0  # Force dep

    admm = ADMMSolver(
        machine="cuda",
        dtype=torch.float32,
        resid_norm=2,
        safe_mode=False,
        scale_dual_residuals=True,
        relative_rho_angle=False,
        # Battery prox specs
        battery_window=24,
        battery_inner_iterations=10,
        battery_inner_over_relaxation=1.8,
        # Algorithm specs
        num_iterations=1000,
        atol=1e-4,
        rho_power=1.0,
        rho_angle=0.5,
        alpha=1.5,
        # Adaptive rho parameters
        adaptive_rho=True,
        adaptation_frequency=10,
        adaptation_tolerance=2.0,
        tau=1.1,
    )
    return admm,


@app.cell(hide_code=True)
def __(devices, num_days):
    print(
        f"Solving for {num_days*24} hours and {sum(d.num_devices for d in devices)} devices."
    )
    return


@app.cell
def __(admm, net, param0, time_horizon, torch_devices):
    state, history = admm.solve(
        net,
        torch_devices,
        time_horizon,
        parameters=param0,
    )
    return history, state


@app.cell(hide_code=True)
def __(devices, np, state, torch, y_cvx):
    _load_met = -np.sum(y_cvx.power[1][0])
    _total_load = -np.sum(devices[1].min_power * devices[1].nominal_capacity)
    _admm_load_met = -torch.sum(state.power[1][0])

    print(f"CVX Load Met: {_load_met:.1f} / {_total_load:.1f}")
    print(f"ADMM Load Met: {_admm_load_met:.1f} / {_total_load:.1f}")
    return


@app.cell
def __(admm, history, plot_convergence):
    plot_convergence(history, eps_pd=admm.primal_tol, fstar=None)  # y_cvx.problem.value)
    return


@app.cell
def __(state, torch):
    torch.mean(torch.abs(state.avg_power)) * 100.0
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


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"## Solve Part 2")
    return


@app.cell
def __(deepcopy, param0, torch):
    param1 = deepcopy(param0)

    for _p in param1:
        for k, v in _p.items():
            _p[k] = v.clone().detach() + torch.rand(v.shape, device="cuda") * 0.010
    return k, param1, v


@app.cell(disabled=True)
def __(admm, net, param1, state, time_horizon, torch_devices):
    state1, history1 = admm.solve(
        net,
        torch_devices,
        time_horizon,
        parameters=param1,
        initial_state=state.copy(),
    )
    return history1, state1


@app.cell
def __(admm, history1, np, plot_convergence):
    plot_convergence(history1, eps_pd=admm.rtol * np.sqrt(admm.total_terminals))
    return


if __name__ == "__main__":
    app.run()
