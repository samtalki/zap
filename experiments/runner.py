import sys
import pypsa
import wandb
import numpy as np
import pandas as pd
import datetime as dt
import cvxpy as cp

from pathlib import Path
from copy import deepcopy

import zap
import zap.planning.trackers as tr


ZAP_PATH = Path(zap.__file__).parent.parent
DATA_PATH = ZAP_PATH / "data"

PARAMETERS = {
    "generator": (zap.Generator, "nominal_capacity"),
    "dc_line": (zap.DCLine, "nominal_capacity"),
    "ac_line": (zap.ACLine, "nominal_capacity"),
    "battery": (zap.Battery, "power_capacity"),
}

ALGORITHMS = {
    "gradient_descent": zap.planning.GradientDescent,
}


def load_config(config_path):
    # TODO Expand configs
    # TODO Hunt for improperly parsed scientific notation
    pass


def load_dataset(config):
    print("Loading dataset...")

    data = config["data"]

    if data["name"] == "pypsa":
        # Load pypsa file
        csv_dir = f"elec_s_{data['num_nodes']}"
        if data["use_extra_components"]:
            csv_dir += "_ec"

        pn = pypsa.Network()
        pn.import_from_csv_folder(DATA_PATH / "pypsa/western/" / csv_dir)

        # Filter out extra components (battery nodes, links, and stores)
        # TODO

        # Pick dates
        start_hour = dt.datetime(2019, 1, 2, 0) + dt.timedelta(hours=data["start_hour"])
        dates = pd.date_range(
            start_hour,
            start_hour + dt.timedelta(hours=data["num_hours"]),
            freq="1h",
            inclusive="left",
        )

        # Build zap network
        net, devices = zap.importers.load_pypsa_network(pn, dates, **data["args"])

        if (not data["use_batteries"]) or (data["num_hours"] == 1):
            devices = [d for d in devices if type(d) != zap.Battery]

        if data["add_ground"]:
            ground = zap.Ground(
                num_nodes=net.num_nodes, terminal=np.array([0]), voltage=np.array([0.0])
            )
            devices += [ground]

    else:
        raise ValueError("Unknown dataset")

    return {
        "net": net,
        "devices": devices,
    }


def setup_problem(data, config):
    print("Building planning problem...")

    cfg = config["problem"]
    net, devices = data["net"], data["devices"]

    # Setup parameters
    parameter_names = {}
    for dev in cfg["parameters"]:
        d_type, param_field = PARAMETERS[dev]
        d_index = device_index(devices, d_type)

        if d_index is not None:
            parameter_names[dev] = d_index, param_field
        else:
            print(f"Warning: device {dev} not found in devices. Will not be expanded.")

    # Setup layer
    layer = zap.DispatchLayer(
        net,
        devices,
        parameter_names=parameter_names,
        time_horizon=devices[0].time_horizon,
        solver=cp.MOSEK,
        solver_kwargs={"verbose": False, "accept_unknown": True},
        add_ground=False,
    )

    # Build objective
    f_cost = cfg["cost_weight"] * zap.planning.DispatchCostObjective(net, devices)
    f_emissions = cfg["emissions_weight"] * zap.planning.EmissionsObjective(devices)
    op_objective = f_cost + f_emissions
    inv_objective = zap.planning.InvestmentObjective(devices, layer)

    # Setup planning problem
    problem = zap.planning.PlanningProblem(
        operation_objective=op_objective,
        investment_objective=inv_objective,
        layer=deepcopy(layer),
        lower_bounds=None,
        upper_bounds=None,
        regularize=cfg["regularize"],
    )

    return {
        "problem": problem,
        "layer": layer,
    }


def solve_relaxed_problem(problem, config):
    problem = problem["problem"]

    if not config["relaxation"]["should_solve"]:
        print("Skipping relaxation...")
        return None

    else:
        relaxation = zap.planning.RelaxedPlanningProblem(
            problem,
            max_price=config["relaxation"]["price_bound"],
            inf_value=config["relaxation"]["inf_value"],
            sd_tolerance=1e-3,
        )

        relaxed_parameters, data = relaxation.solve()

        return {
            "relaxation": relaxation,
            "relaxed_parameters": relaxed_parameters,
            "data": data,
            "lower_bound": data["problem"].value,
        }


def solve_problem(problem, relaxation, config):
    print("Solving problem...")
    opt_config = config["optimizer"]
    problem: zap.planning.PlanningProblem = problem["problem"]

    # Construct algorithm
    alg = ALGORITHMS[opt_config["name"]](**opt_config["args"])

    # Setup wandb
    if config["system"]["use_wandb"]:
        wandb.init(project="zap", config=config)
        logger = wandb
    else:
        logger = None

    # Initialize
    if relaxation is not None and opt_config["initial_state"] == "relaxation":
        print("Initializing with relaxation solution.")
        initial_state = deepcopy(relaxation["relaxed_parameters"])

    else:
        print("Initializing with initial parameters (no investment).")
        initial_state = None

    # Solve
    parameters, history = problem.solve(
        num_iterations=opt_config["num_iterations"],
        algorithm=alg,
        trackers=tr.DEFAULT_TRACKERS,
        initial_state=initial_state,
        wandb=logger,
        log_wandb_every=config["system"]["log_wandb_every"],
        lower_bound=relaxation["lower_bound"] if relaxation is not None else None,
    )

    if config["system"]["use_wandb"]:
        wandb.finish()

    return {
        "parameters": parameters,
        "history": history,
    }


def save_results(relaxation, results, config):
    pass


def run_experiment(config):
    # Load data and formulate problem
    data = load_dataset(config)
    problem = setup_problem(data, config)

    # Solve relaxation and original problem
    relaxation = solve_relaxed_problem(problem, config)
    results = solve_problem(problem, relaxation, config)

    save_results(relaxation, results, config)


# ====
# Utility Functions
# ====


def device_index(devices, kind):
    if not any(isinstance(d, kind) for d in devices):
        return None

    return next(i for i, d in enumerate(devices) if isinstance(d, kind))


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    run_experiment(config)
