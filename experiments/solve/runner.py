import sys
import platform
import numpy as np

from pathlib import Path

import zap.util

sys.path.append(Path(__file__).parent.parent)

import zap
from experiments.plan import runner

from zap import DispatchLayer
from zap.admm import ADMMLayer, ADMMSolver

SOLVERS = ["admm", "cvxpy"]


def load_config(path):
    return runner.load_config(path)


def expand_config(config):
    return runner.expand_config(config)


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
    solver=None,
    config=None,
):
    assert solver in SOLVERS

    # Select arguments from config
    args_name = solver + "_args"
    args = config[args_name] if args_name in config else {}

    # Convert cases to torch if needed (in place operation)
    if solver == "admm":
        print("Torchifying cases...")
        args["dtype"] = runner.TORCH_DTYPES[args["dtype"]]

        machine = args["machine"]
        dtype = args["dtype"]

        for i in range(len(time_cases)):
            time_cases[i] = [d.torchify(machine=machine, dtype=dtype) for d in time_cases[i]]

    # Expand inner arguments
    arg_list = runner.expand_config(args, key="sweep")

    # Build a layer for each case and each argument setting
    layers = []
    for arg in arg_list:
        del arg["index"]
        for case in time_cases:
            layers += [build_layer(net, case, parameters, solver, arg)]

    print(f"Initialized {len(layers)} layers with {solver} solver.")
    return layers


def build_layer(net, case, parameters, solver, args):
    time_horizon = np.max([d.time_horizon for d in case])

    if solver == "admm":
        return ADMMLayer(
            net,
            case,
            parameter_names=parameters,
            time_horizon=time_horizon,
            solver=ADMMSolver(**args),
            adapt_rho=False,
            warm_start=False,
        )

    if solver == "cvxpy":
        return DispatchLayer(
            net,
            case,
            parameter_names=parameters,
            time_horizon=time_horizon,
            **args,
        )


def solve_problem(layers, param_cases):
    i = 0

    for layer in layers:
        for theta in param_cases:
            print(f"Solving case {i}.")

            # Convert layer to torch, if needed
            if isinstance(layer, ADMMLayer):
                theta = zap.util.torchify(
                    theta, machine=layer.solver.machine, dtype=layer.solver.dtype
                )

            layer(**theta)
            i += 1

    print("Solved all cases.")


def run_experiment(config):
    data = load_network(**config["data"])
    print("\n\n\n")

    net, devices = data["net"], data["devices"]
    time_cases, capacity_cases, parameters = load_parameter_set(
        net, devices, **config["parameters"]
    )
    layers = setup_layers(net, time_cases, parameters, solver=config["solver"], config=config)
    print("\n\n\n")

    solve_problem(layers, capacity_cases)


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
