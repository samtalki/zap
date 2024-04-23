import numpy as np
import pandas as pd
import datetime as dt
import cvxpy as cp
import sys
import pypsa
import wandb
import yaml
import json

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


def get_results_path(config_name):
    return DATA_PATH / "results" / config_name


def load_config(path):
    path = Path(path)

    # Open config
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Tag config from name
    config_name = Path(path).name.split(".")[0] + "_000"

    # Check if name has already been used
    while get_results_path(config_name).exists():
        config_name = config_name[:-3] + f"{int(config_name[-3:]) + 1:03d}"

    config["name"] = config_name

    # TODO Expand configs
    # TODO Hunt for improperly parsed scientific notation

    return config


def load_dataset(config):
    print("Loading dataset...")

    data = config["data"]

    if data["name"] == "pypsa":
        net, devices = setup_pypysa_dataset(data)

    else:
        raise ValueError("Unknown dataset")

    return {
        "net": net,
        "devices": devices,
    }


def setup_pypysa_dataset(data):
    # Load pypsa file
    csv_dir = f"elec_s_{data['num_nodes']}"
    if data["use_extra_components"]:
        csv_dir += "_ec"

    pn = pypsa.Network()
    pn.import_from_csv_folder(DATA_PATH / "pypsa/western/" / csv_dir)

    # Filter out extra components (battery nodes, links, and stores)
    pn.buses = pn.buses[~pn.buses.index.str.contains("battery")]
    pn.links = pn.links[~pn.links.index.str.contains("battery")]

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

    return net, devices


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
        print("Solving relaxation...")
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


def solve_problem(problem_data, relaxation, config):
    print("Solving problem...")
    opt_config = config["optimizer"]
    problem: zap.planning.PlanningProblem = problem_data["problem"]

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
        trackers=tr.DEFAULT_TRACKERS + [tr.GRAD],
        initial_state=initial_state,
        wandb=logger,
        log_wandb_every=config["system"]["log_wandb_every"],
        lower_bound=relaxation["lower_bound"] if relaxation is not None else None,
        extra_wandb_trackers=get_wandb_trackers(problem_data, relaxation, config),
    )

    return {
        "parameters": parameters,
        "history": history,
    }


def save_results(relaxation, results, config):
    # Pick a file name
    results_path = get_results_path(config["name"])
    results_path.mkdir(parents=True, exist_ok=False)

    # Save relaxation parameters
    relax_params = {k: v.ravel().tolist() for k, v in relaxation["relaxed_parameters"].items()}
    with open(results_path / "relaxed.json", "w") as f:
        json.dump(relax_params, f)

    # Save final parameters
    final_params = {k: v.ravel().tolist() for k, v in results["parameters"].items()}
    with open(results_path / "optimized.json", "w") as f:
        json.dump(final_params, f)

    return None


def run_experiment(config):
    # Load data and formulate problem
    data = load_dataset(config)
    problem = setup_problem(data, config)

    # Solve relaxation and original problem
    relaxation = solve_relaxed_problem(problem, config)
    results = solve_problem(problem, relaxation, config)

    save_results(relaxation, results, config)

    if config["system"]["use_wandb"]:
        wandb.finish()

    return None


# ====
# Utility Functions
# ====


def device_index(devices, kind):
    if not any(isinstance(d, kind) for d in devices):
        return None

    return next(i for i, d in enumerate(devices) if isinstance(d, kind))


def get_wandb_trackers(problem_data, relaxation, config):
    problem, layer = problem_data["problem"], problem_data["layer"]

    carbon_objective = zap.planning.EmissionsObjective(layer.devices)
    cost_objective = zap.planning.DispatchCostObjective(layer.network, layer.devices)

    def emissions_tracker(J, grad, state, last_state, problem):
        return carbon_objective(problem.state, parameters=layer.setup_parameters(**state))

    def cost_tracker(J, grad, state, last_state, problem):
        return cost_objective(problem.state, parameters=layer.setup_parameters(**state))

    lower_bound = relaxation["lower_bound"] if relaxation is not None else 1.0
    true_relax_cost = relaxation["data"]["problem"].value if relaxation is not None else np.inf
    relax_solve_time = (
        relaxation["data"]["problem"].solver_stats.solve_time if relaxation is not None else 0.0
    )

    # Trackers
    return {
        "emissions": emissions_tracker,
        "fuel_costs": cost_tracker,
        "inv_cost": lambda *args: problem.inv_cost.item(),
        "op_cost": lambda *args: problem.op_cost.item(),
        "lower_bound": lambda *args: lower_bound,
        "true_relaxation_cost": lambda *args: true_relax_cost,
        "relaxation_solve_time": lambda *args: relax_solve_time,
    }


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    run_experiment(config)
