import pypsa
import datetime as dt
import pandas as pd
import cvxpy as cp
import numpy as np
import time

from copy import deepcopy

import zap
from zap import DispatchLayer

import multiprocessing as mp
import concurrent.futures as con


DEFAULT_PYPSA_KWARGS = {
    "marginal_load_value": 1000.0,
    "load_cost_perturbation": 10.0,
    "generator_cost_perturbation": 1.0,
}

PYP_NET = pypsa.Network(f"~/pypsa-usa/workflow/resources/western/elec_s_100.nc")


def load_pypsa_network(
    time_horizon=1,
    num_nodes=100,
    start_date=dt.datetime(2019, 1, 2, 0),
    exclude_batteries=False,
    **pypsa_kwargs,
):
    all_kwargs = deepcopy(DEFAULT_PYPSA_KWARGS)
    all_kwargs.update(pypsa_kwargs)
    print(all_kwargs)

    dates = pd.date_range(
        start_date,
        start_date + dt.timedelta(hours=time_horizon),
        freq="1h",
        inclusive="left",
    )

    net, devices = zap.importers.load_pypsa_network(PYP_NET, dates, **all_kwargs)
    if exclude_batteries:
        devices = devices[:-1]

    return net, devices, time_horizon


def dev_index(devices, tp):
    indices = [i for i, dev in enumerate(devices) if isinstance(dev, tp)]
    if len(indices) > 0:
        return indices[0]
    else:
        return None


def make_layer(
    net,
    devices,
    time_horizon,
    use_gens=True,
    use_lines=True,
    solver=cp.MOSEK,
    solver_opts={},
):
    gen_index = dev_index(devices, zap.Generator)
    line_index = dev_index(devices, zap.ACLine)

    parameter_names = {}
    if use_gens:
        parameter_names["generator_capacity"] = (gen_index, "nominal_capacity")
    if use_lines:
        parameter_names["line_capacity"] = (line_index, "nominal_capacity")

    layer = DispatchLayer(
        net,
        devices,
        parameter_names=parameter_names,
        time_horizon=time_horizon,
        solver=solver,
        solver_kwargs=solver_opts,
    )

    parameter_values = {}
    for name, (index, attr) in parameter_names.items():
        parameter_values[name] = getattr(devices[index], attr)

    return layer, parameter_values


def run_little_simulation(case):
    _F, _theta = make_layer(
        *case,
        use_lines=True,
        solver=cp.MOSEK,
        solver_opts={"verbose": False, "accept_unknown": True},
    )
    _y = _F(**_theta)
    return _y


def f(x):
    return x * x


if __name__ == "__main__":
    num_workers = 1
    num_cases = 16
    time_horizon = 24

    cases = [
        load_pypsa_network(
            time_horizon=time_horizon,
            start_date=dt.datetime(2019, 6, 1, 0) + dt.timedelta(hours=time_horizon * day),
        )
        for day in range(num_cases)
    ]
    print("Done")

    start = time.time()
    # with mp.Pool(processes=num_workers) as pool:
    #     results = pool.map(run_little_simulation, cases)

    with con.ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = pool.map(run_little_simulation, cases)

    results = [r for r in results]

    runtime = time.time() - start
    print("\n\nRuntime: ", runtime, "\n\n")
    print(type(results[0].power[0][0]))

    # jobs = []
    # for i in range(0, num_workers):
    #     process = mp.Process(target=run_little_simulation, args=(cases[i],))
    #     jobs.append(process)

    # # Start the processes
    # for j in jobs:
    #     j.start()

    # # Ensure all of the processes have finished
    # for j in jobs:
    #     j.join()
