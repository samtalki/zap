import marimo

__generated_with = "0.9.1"
app = marimo.App()


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
    return (zap,)


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_theme()
    return plt, seaborn


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"## Setup Cases")
    return


@app.cell(hide_code=True)
def __(pypsa):
    network_cache = {}


    def load_network(num_nodes):
        if num_nodes not in network_cache.keys():
            pn = pypsa.Network()
            pn.import_from_csv_folder(f"data/pypsa/western/load_medium/elec_s_{num_nodes}")
            network_cache[num_nodes] = pn

        return network_cache[num_nodes]
    return load_network, network_cache


@app.cell(hide_code=True)
def __(dt, np, pd, zap):
    def load_case(
        pn,
        time_horizon=1,
        start_date=dt.datetime(2019, 8, 9, 7),
        exclude_batteries=False,
        power_unit=1000.0,
        cost_unit=10.0,
        marginal_load_value=500.0,
        load_cost_perturbation=10.0,
        generator_cost_perturbation=1.0,
        scale_load=0.6,
        scale_generator_capacity_factor=0.7,
        scale_line_capacity_factor=0.7,
        drop_empty_generators=False,
        expand_empty_generators=0.5,
        battery_discharge_cost=1.0,
        battery_init_soc=0.0,
        battery_final_soc=0.0,
    ):
        dates = pd.date_range(
            start_date,
            start_date + dt.timedelta(hours=time_horizon),
            freq="1h",
            inclusive="left",
        )

        net, devices = zap.importers.load_pypsa_network(
            pn,
            dates,
            power_unit=power_unit,
            cost_unit=cost_unit,
            marginal_load_value=marginal_load_value,
            load_cost_perturbation=load_cost_perturbation,
            generator_cost_perturbation=generator_cost_perturbation,
            scale_load=scale_load,
            scale_generator_capacity_factor=scale_generator_capacity_factor,
            scale_line_capacity_factor=scale_line_capacity_factor,
            drop_empty_generators=drop_empty_generators,
            expand_empty_generators=expand_empty_generators,
            battery_discharge_cost=battery_discharge_cost,
            battery_init_soc=battery_init_soc,
            battery_final_soc=battery_final_soc,
        )
        if exclude_batteries:
            devices = devices[:-1]

        _ground = zap.Ground(
            num_nodes=net.num_nodes,
            terminal=np.array([0]),
            voltage=np.array([0.0]),
        )
        devices += [_ground]

        return net, devices, time_horizon
    return (load_case,)


@app.cell
def __():
    time_horizon = 24
    return (time_horizon,)


@app.cell
def __():
    num_nodes = [100, 200, 240, 500, 1000, 4000]
    return (num_nodes,)


@app.cell
def __(load_network, num_nodes):
    import warnings

    with warnings.catch_warnings():
        networks = [load_network(n) for n in num_nodes]
    return networks, warnings


@app.cell(hide_code=True)
def __(load_case, networks, time_horizon):
    cases = [load_case(pn, time_horizon=time_horizon) for pn in networks]
    return (cases,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Solve Baselines""")
    return


@app.cell
def __(cp):
    def solve_baseline(net, devices, time_horizon):
        print(f"Solving problem with {net.num_nodes} nodes.")
        return net.dispatch(
            devices,
            add_ground=False,
            solver=cp.MOSEK,
            solver_kwargs={"verbose": True}
        )
    return (solve_baseline,)


@app.cell
def __(cases, solve_baseline):
    baseline_solves = [solve_baseline(*c) for c in cases]
    return (baseline_solves,)


@app.cell
def __(baseline_solves, np):
    np.sum(baseline_solves[-1].power[1][0])
    return


@app.cell
def __(cases, np):
    _l = cases[-1][1][1]
    np.sum(_l.nominal_capacity * _l.min_power)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Solve ADMM""")
    return


@app.cell
def __():
    from zap.admm import ADMMSolver
    return (ADMMSolver,)


@app.cell
def __(torch):
    machine, dtype = "cuda", torch.float32
    return dtype, machine


@app.cell
def __(ADMMSolver, dtype, machine):
    def solve_case(
        net,
        devices,
        time_horizon,
        num_iterations=2500,
        resid_norm=2,
        atol=1e-8,
        rtol=1e-4,
        rho_power=1.0,
        rho_angle=1.0,
        adaptive_rho=True,
        verbose=False,
        **kwargs,
    ):
        print(f"Solving problem with {net.num_nodes} nodes.")
        admm = ADMMSolver(
            num_iterations,
            rho_power,
            machine=machine,
            dtype=dtype,
            resid_norm=resid_norm,
            atol=atol,
            rtol=rtol,
            rho_angle=rho_angle,
            adaptive_rho=adaptive_rho,
            verbose=verbose,
            **kwargs,
        )
        state, history = admm.solve(net, devices, time_horizon)
        return state, history, admm
    return (solve_case,)


@app.cell
def __(dtype, machine, torch):
    def torchify_case(net, devices, time_horizon):
        torch.cuda.empty_cache()
        torch_devices = [d.torchify(machine=machine, dtype=dtype) for d in devices]

        return net, torch_devices, time_horizon
    return (torchify_case,)


@app.cell
def __(solve_case, torched_cases):
    admm_solves = [solve_case(*c) for c in torched_cases]
    return (admm_solves,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Analyze Results""")
    return


@app.cell(hide_code=True)
def __(np, plt):
    def plot_convergence(state, hist, admm, fstar=None):
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

        print(f"Primal Resid:\t\t\t {hist.power[-1] + hist.phase[-1]:.4f}")
        print(f"Dual Resid:\t\t\t\t {hist.dual_power[-1] + hist.dual_phase[-1]:.4f}")

        if fstar is None:
            print(f"Objective:\t\t\t {hist.objective[-1]:.2f}")
        else:
            print(f"Objective (Optimal):\t {hist.objective[-1]:.2f}\t ({fstar:.2f})")
            print(
                f"Objective Gap:\t\t\t {100.0*np.abs(hist.objective[-1] - fstar) / np.abs(fstar):.2f} %"
            )

        admm_num_iters = len(hist.power)

        ax = axes[0]
        ax.hlines(admm.primal_tol, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.power, label="power")
        ax.plot(hist.phase, label="angle")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title("primal residuals")

        ax = axes[1]
        ax.hlines(admm.dual_tol, xmin=0, xmax=admm_num_iters, color="black", zorder=-100)
        ax.plot(hist.dual_power, label="power")
        ax.plot(hist.dual_phase, label="angle")
        ax.set_yscale("log")
        ax.set_title("dual residuals")

        ax = axes[2]
        ax.plot(np.array(hist.objective))
        ax.set_title("f")
        if fstar is not None:
            ax.hlines(fstar, xmin=0, xmax=len(hist.objective), color="black")

        fig.tight_layout()
        return fig
    return (plot_convergence,)


@app.cell
def __(np, torch):
    def load_statistics(case, solution, baseline):
        state, history, admm = solution
        _, devices, _ = case

        total_load = -np.sum(devices[1].min_power * devices[1].nominal_capacity)
        load_met = -np.sum(baseline.power[1][0])
        admm_load_met = -torch.sum(state.power[1][0])

        print(
            f"ADMM (CVX) Load Met: {admm_load_met:.1f} ({load_met:.1f}) / {total_load:.1f}"
        )
    return (load_statistics,)


@app.cell(hide_code=True)
def __(np, zap):
    def get_primal_resid(solve):
        state, hist, admm = solve
        return np.sqrt(hist.power[-1] ** 2 + hist.phase[-1] ** 2)


    def get_dual_resid(solve):
        state, hist, admm = solve
        return np.sqrt(hist.dual_power[-1] ** 2 + hist.dual_phase[-1] ** 2)


    def get_true_objective(baseline):
        return baseline.problem.value


    def get_admm_objective(solve):
        state, hist, admm = solve
        return hist.objective[-1]


    def norm_p(solve, p):
        state, _, _ = solve
        power = zap.admm.util.nested_norm(state.power, p).item() ** p
        theta = zap.admm.util.nested_norm(state.phase, p).item() ** p

        return power + theta
    return (
        get_admm_objective,
        get_dual_resid,
        get_primal_resid,
        get_true_objective,
        norm_p,
    )


@app.cell
def __(
    get_admm_objective,
    get_dual_resid,
    get_primal_resid,
    get_true_objective,
    norm_p,
    np,
    pd,
    time_horizon,
):
    def aggregate_stats(cases, baselines, solutions):
        # Case data
        num_nodes = [c[0].num_nodes for c in cases]
        num_terminals = [s[2].total_terminals / time_horizon for s in solutions]

        # Residuals
        r_primal = [get_primal_resid(s) for s in solutions]
        r_dual = [get_dual_resid(s) for s in solutions]

        # Objectives
        f_opt = [get_true_objective(b) for b in baselines]
        f_admm = [get_admm_objective(s) for s in solutions]

        df = pd.DataFrame(
            {
                "num_nodes": num_nodes,
                "num_terminals": num_terminals,
                "r_primal": r_primal,
                "r_dual": r_dual,
                "f_opt": f_opt,
                "f_admm": f_admm,
            }
        )

        df["subopt"] = np.abs(df.f_admm - df.f_opt) / df.f_opt

        # Variable scaling
        df["sq_norm_x"] = [norm_p(s, 2) for s in solutions]
        df["norm_x"] = np.sqrt(df["sq_norm_x"])
        df["sq_norm_x_over_j"] = df.sq_norm_x / df.num_terminals

        # Tolerance
        df["primal_tol"] = [s[2].primal_tol for s in solutions]
        df["dual_tol"] = [s[2].dual_tol for s in solutions]

        return df
    return (aggregate_stats,)


@app.cell
def __(admm_solves, aggregate_stats, baseline_solves, cases):
    aggregate_stats(cases, baseline_solves, admm_solves)
    return


@app.cell
def __(admm_solves, baseline_solves, plot_convergence):
    _i = 5
    plot_convergence(*admm_solves[_i], fstar=baseline_solves[_i].problem.value)
    return


@app.cell
def __(admm_solves, baseline_solves, cases, load_statistics):
    _i = 5
    load_statistics(cases[_i], admm_solves[_i], baseline_solves[_i])
    return


@app.cell
def __():
    # torch.mean(torch.abs(state.avg_power)) * 100.0
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Appendix""")
    return


@app.cell
def __(cases, torchify_case):
    torched_cases = [torchify_case(*c) for c in cases]
    return (torched_cases,)


if __name__ == "__main__":
    app.run()
