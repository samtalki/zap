import sys
import platform
import pickle
import time
import numpy as np
import scipy.sparse as sp
import torch

from pathlib import Path

import zap.util

sys.path.append(Path(__file__).parent.parent)

import zap
from experiments.plan import runner

from zap import DispatchLayer
from zap.admm import ADMMLayer, ADMMSolver

SOLVERS = ["admm", "cvxpy"]


def datadir(*args):
    return runner.datadir("solver", *args)


def load_config(path):
    return runner.load_config(path)


def expand_config(config, key="expand"):
    return runner.expand_config(config, key=key)


def load_network(**kwargs):
    """Load the network and device data."""
    return runner.load_dataset(**kwargs)


def load_parameter_set(
    net,
    devices,
    parameter_types=["generator", "dc_line", "ac_line", "battery"],
    hours_per_scenario=0,
    capacities=["base"],
):
    """Load the network parameter and split problem up into cases."""
    time_horizon = np.max([d.time_horizon for d in devices])

    # Split problem up by dates
    if hours_per_scenario == 0:
        hours_per_scenario = time_horizon
    assert time_horizon % hours_per_scenario == 0

    print(
        f"Solving {time_horizon / hours_per_scenario} problems with {hours_per_scenario} hours each."
    )

    scenarios = [
        range(i, i + hours_per_scenario) for i in range(0, time_horizon, hours_per_scenario)
    ]
    time_cases = [[d.sample_time(s, time_horizon) for d in devices] for s in scenarios]

    # Setup parameters
    parameter_names = {}
    for dev in parameter_types:
        d_type, param_field = runner.PARAMETERS[dev]
        d_index = runner.device_index(devices, d_type)

        if d_index is not None:
            parameter_names[dev] = d_index, param_field
        else:
            print(f"Warning: device {dev} not found in devices. Will not be expanded.")

    # Build capacities parameters
    toy_layer = zap.DispatchLayer(net, devices, parameter_names, hours_per_scenario)
    capacity_cases = [parse_capacity(c, toy_layer) for c in capacities]

    return time_cases, capacity_cases, parameter_names


def parse_capacity(capacity, layer):
    initial_params = layer.initialize_parameters()

    if capacity == "base":
        return initial_params

    # Existing file
    elif isinstance(capacity, str):
        return NotImplementedError

    # Uniform scaling
    elif isinstance(capacity, float):
        return {k: v * capacity for k, v in initial_params.items()}


def setup_layers(
    net,
    time_cases,
    parameters,
    battery_window=0,
    num_contingencies=0,
    solver=None,
    config=None,
):
    assert solver in SOLVERS

    # Select arguments from config
    args_name = solver + "_args"
    args = config[args_name] if args_name in config else {}

    # Add contingencies, if needed
    if num_contingencies > 0:
        print(f"Adding {num_contingencies} contingencies.")
        contingency_device = runner.device_index(time_cases[0], zap.ACLine)
        print(f"Contingency device: {contingency_device}")

        if num_contingencies > time_cases[0][contingency_device].num_devices:
            num_contingencies = time_cases[0][contingency_device].num_devices
            print(
                f"Warning: not enough devices for contingencies. Reducing to max of {num_contingencies}."
            )

        # Build contingency mask
        cmask = sp.lil_matrix((num_contingencies, time_cases[0][contingency_device].num_devices))
        for c in range(num_contingencies):
            cmask[c, c] = 1.0
        cmask = cmask.tocsr()

    else:
        print("Solving without contingencies.")
        contingency_device = None
        cmask = None

    # Convert cases to torch if needed (in place operation)
    if solver == "admm":
        print("Torchifying cases...")
        args["dtype"] = runner.TORCH_DTYPES[args["dtype"]]

        machine = args["machine"]
        dtype = args["dtype"]

        for i in range(len(time_cases)):
            time_cases[i] = [d.torchify(machine=machine, dtype=dtype) for d in time_cases[i]]

        if cmask is not None:
            cmask = torch.tensor(cmask.todense(), device=machine, dtype=dtype)
            cmask = torch.vstack(
                [
                    torch.zeros(cmask.shape[1], device=machine, dtype=dtype),  # Base case
                    cmask,  # Contingencies
                ]
            )

    # Expand inner arguments
    arg_list = runner.expand_config(args, key="sweep")

    # Build a layer for each case and each argument setting
    layers = []
    for arg in arg_list:
        del arg["index"]
        for case in time_cases:
            layers += [
                build_layer(
                    net,
                    case,
                    parameters,
                    solver,
                    arg,
                    battery_window,
                    num_contingencies,
                    contingency_device,
                    cmask,
                )
            ]

    print(f"Initialized {len(layers)} layers with {solver} solver.")
    return layers


def build_layer(
    net,
    case,
    parameters,
    solver,
    args,
    battery_window,
    num_contingencies,
    contingency_device,
    cmask,
):
    time_horizon = np.max([d.time_horizon for d in case])
    if battery_window == 0:
        battery_window = time_horizon
        print(f"Setting battery window to {battery_window}.")

    assert time_horizon % battery_window == 0

    if solver == "admm":
        return ADMMLayer(
            net,
            case,
            parameter_names=parameters,
            time_horizon=time_horizon,
            solver=ADMMSolver(battery_window=battery_window, **args),
            adapt_rho=False,
            warm_start=False,
            verbose=True,
            num_contingencies=num_contingencies,
            contingency_device=contingency_device,
            contingency_mask=cmask,
        )

    if solver == "cvxpy":
        # Sample time by battery window
        windows = [range(i, i + battery_window) for i in range(0, time_horizon, battery_window)]
        subcases = [[d.sample_time(s, time_horizon) for d in case] for s in windows]
        print(f"CVXPY splitting into {len(subcases)} individual problems.")
        return [
            DispatchLayer(
                net,
                c,
                parameter_names=parameters,
                time_horizon=battery_window,
                num_contingencies=num_contingencies,
                contingency_device=contingency_device,
                contingency_mask=cmask,
                **args,
            )
            for c in subcases
        ]


def solve_problem(layers, param_cases):
    i = 0

    ys = []
    solver_data = []

    for i_theta, theta in enumerate(param_cases):
        ys += [[]]
        solver_data += [[]]

        for i_layer, layer in enumerate(layers):
            print(f"Solving case {i}.")

            # Convert layer to torch, if needed
            if isinstance(layer, ADMMLayer):
                theta = zap.util.torchify(
                    theta, machine=layer.solver.machine, dtype=layer.solver.dtype
                )

            t0 = time.time()
            if isinstance(layer, list):
                y = [layer[i](**theta) for i in range(len(layer))]
            else:
                rho_power, rho_angle = layer.solver.rho_power, layer.solver.rho_angle
                y = layer(**theta)
                # Need to reset these after each solve
                layer.solver.rho_power = rho_power
                layer.solver.rho_angle = rho_angle

            solve_time = time.time() - t0
            print(f"Solved in {solve_time:.2f} seconds.")

            i += 1
            if isinstance(layer, ADMMLayer):
                solver_data[i_theta] += [
                    {
                        "time": solve_time,
                        "history": layer.history,
                        "iteration": layer.solver.iteration,
                        "converged": layer.solver.converged,
                        "num_dc_terminals": layer.solver.num_dc_terminals,
                        "num_ac_terminals": layer.solver.num_ac_terminals,
                        "total_terminals": layer.solver.total_terminals,
                        "primal_tol_power": layer.solver.primal_tol_power,
                        "primal_tol_angle": layer.solver.primal_tol_angle,
                        "dual_tol_power": layer.solver.dual_tol_power,
                        "dual_tol_angle": layer.solver.dual_tol_angle,
                        "primal_tol": layer.solver.primal_tol,
                        "dual_tol": layer.solver.dual_tol,
                        "rho_power": layer.solver.rho_power,
                        "rho_angle": layer.solver.rho_angle,
                    }
                ]
                ys += [(y, layer.state)]

            else:
                solver_data[i_theta] += [
                    {"time": solve_time, "problem_data": [cvxpy_data(yi) for yi in y]}
                ]
                for yi in y:
                    yi.problem = None  # This got moved to solver_data

                ys += y

    print("Solved all cases.")
    return ys, solver_data


def cvxpy_data(y):
    prob = y.problem
    return {
        "status": prob.status,
        "value": prob.value,
        "solver_stats": prob.solver_stats,
        "compilation_time": prob.compilation_time,
    }


def save_results(ys, solver_data, time_cases, capacity_cases, config):
    results_path = get_results_path(config["id"], config.get("index", None))
    results_path.mkdir(parents=True, exist_ok=True)

    # Save ys
    with open(results_path / "ys.pkl", "wb") as f:
        pickle.dump(ys, f)

    # Save solver ouptuts
    with open(results_path / "solver_data.pkl", "wb") as f:
        pickle.dump(solver_data, f)

    # Save times and capacities
    cases = {"time": len(time_cases), "capacity": capacity_cases}
    with open(results_path / "cases.pkl", "wb") as f:
        pickle.dump(cases, f)


def get_results_path(config_name, index=None):
    if index is None:
        return datadir(config_name)
    else:
        return datadir(config_name, f"{index:03d}")


def run_experiment(config):
    data = load_network(**config["data"])
    print("\n\n\n")

    net, devices = data["net"], data["devices"]
    time_cases, capacity_cases, parameters = load_parameter_set(
        net, devices, **config["parameters"]
    )
    layers = setup_layers(
        net,
        time_cases,
        parameters,
        solver=config["solver"],
        battery_window=config["battery_window"],
        num_contingencies=config.get("num_contingencies", 0),
        config=config,
    )
    print("\n\n\n")

    ys, solver_data = solve_problem(layers, capacity_cases)
    save_results(ys, solver_data, time_cases, capacity_cases, config)


if __name__ == "__main__":
    config_path = sys.argv[1]

    if len(sys.argv) > 2:
        config_num = int(sys.argv[2])
    else:
        config_num = 0

    config = expand_config(load_config(config_path))[config_num]

    print(platform.architecture())
    print(platform.machine())
    print(platform.platform())
    print(platform.processor())
    print(platform.system())
    print(platform.version())
    print(platform.uname())
    print(platform.python_version())
    print("\n\n\n")

    run_experiment(config)
