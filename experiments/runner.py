import numpy as np
import pandas as pd
import datetime as dt
import cvxpy as cp
import ray
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

TOTAL_PYPSA_HOUR = 8760 - 48
PYPSA_START_DAY = dt.datetime(2019, 1, 2, 0)
PYPSA_DEFAULT_ARGS = {
    "power_unit": 1.0e3,
    "cost_unit": 10.0,
    "marginal_load_value": 500.0,
    "scale_load": 1.5,
    "scale_generator_capacity_factor": 0.8,
    "scale_line_capacity_factor": 0.8,
    "carbon_tax": 0.0,
    "generator_cost_perturbation": 1.0,
    "load_cost_perturbation": 10.0,
    "drop_empty_generators": False,
    "expand_empty_generators": 0.5,
}


def expand_config(config: dict) -> list[dict]:
    configs = _expand_config(config)

    # Tag every config with an index
    for i, c in enumerate(configs):
        c["index"] = i

    return configs


def _expand_config(config: dict) -> list[dict]:
    """Expand config with multiple values for a single parameter.

    Lists of the form `param: [expand, arg1, arg2, ..., argn]` will be expanded into
    `n` distinct configurations.
    """
    config = deepcopy(config)

    # Expand sub-dictionaries
    for k, v in config.items():
        if isinstance(v, dict):
            v = _expand_config(v)
            if len(v) == 1:
                config[k] = v[0]
            else:
                config[k] = ["expand"] + v

    # Expand lists
    for k, v in config.items():
        if isinstance(v, list) and len(v) > 0 and v[0] == "expand":
            data: list = v[1:]
            assert isinstance(data, list)

            expanded_configs = [deepcopy(config) for _ in range(len(data))]

            for i, d in enumerate(data):
                expanded_configs[i][k] = d

            return sum([_expand_config(e) for e in expanded_configs], start=[])  # Join lists

    # No expansion happened, just return a singleton list
    return [config]


def load_config(path):
    path = Path(path)

    if not path.exists():
        path = ZAP_PATH / path

    path = path.absolute()
    assert path.exists()
    print(path)

    # Open config
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Tag config from name
    config_short_name = Path(path).name.split(".")[0]
    config_id = config_short_name  # + "_000"

    # # Check if name has already been used
    # while get_results_path(config_id).exists():
    #     config_id = config_id[:-3] + f"{int(config_id[-3:]) + 1:03d}"

    config["name"] = config_short_name
    config["id"] = config_id

    # TODO Expand configs
    # Hunt for improperly parsed scientific notation (bug in Python yaml parser)
    config = parse_scientific_notation(config)

    return config


def load_dataset(name="pypsa", **kwargs):
    print("Loading dataset...")

    print(name)

    if name == "pypsa":
        net, devices = setup_pypysa_dataset(**kwargs)

    else:
        raise ValueError("Unknown dataset")

    return {
        "net": net,
        "devices": devices,
    }


def setup_pypysa_dataset(
    add_ground=True,
    use_batteries=True,
    use_extra_components=True,
    num_nodes=100,
    start_hour="peak_load_day",
    num_hours=4,
    args=PYPSA_DEFAULT_ARGS,
    **kwargs,
):
    # Load pypsa file
    csv_dir = f"elec_s_{num_nodes}"
    if use_extra_components:
        csv_dir += "_ec"

    pn = pypsa.Network()
    pn.import_from_csv_folder(DATA_PATH / "pypsa/western/" / csv_dir)

    # Filter out extra components (battery nodes, links, and stores)
    pn.buses = pn.buses[~pn.buses.index.str.contains("battery")]
    pn.links = pn.links[~pn.links.index.str.contains("battery")]

    # Pick dates
    # Rule 1 - Just a fixed hour of the year
    if isinstance(start_hour, int):
        print("Using a fixed start hour.")
        start_hour = PYPSA_START_DAY + dt.timedelta(hours=start_hour)

    # Rule 2 - Dynamically selected by peak load
    elif start_hour == "peak_load_day":
        print("Finding the peak load day.")
        all_dates = pd.date_range(start=PYPSA_START_DAY, periods=TOTAL_PYPSA_HOUR, freq="1h")
        _, year_devices = zap.importers.load_pypsa_network(pn, all_dates, **args)

        daily_peak_load = get_total_load_curve(year_devices, every=24, reducer=np.max)
        peak_day = np.argmax(daily_peak_load).item()

        start_hour = PYPSA_START_DAY + dt.timedelta(hours=peak_day * 24)

    # Build zap network
    dates = pd.date_range(start_hour, periods=num_hours, freq="1h", inclusive="left")
    print(f"Solving a case with {len(dates)} hours.")
    print(dates)
    net, devices = zap.importers.load_pypsa_network(pn, dates, **args)

    if (not use_batteries) or (num_hours == 1):
        devices = [d for d in devices if type(d) != zap.Battery]

    if add_ground:
        ground = zap.Ground(
            num_nodes=net.num_nodes, terminal=np.array([0]), voltage=np.array([0.0])
        )
        devices += [ground]

    return net, devices


def setup_problem(
    net,
    devices,
    parameters=["generator", "dc_line", "ac_line", "battery"],
    cost_weight=1.0,
    emissions_weight=0.0,
    regularize=0.0,
    stochastic=False,
    hours_per_scenario=1,
):
    print("Building planning problem...")
    time_horizon = np.max([d.time_horizon for d in devices])

    # Drop batteries
    if stochastic and hours_per_scenario == 1:
        devices = [d for d in devices if type(d) != zap.Battery]

    # Setup parameters
    parameter_names = {}
    for dev in parameters:
        d_type, param_field = PARAMETERS[dev]
        d_index = device_index(devices, d_type)

        if d_index is not None:
            parameter_names[dev] = d_index, param_field
        else:
            print(f"Warning: device {dev} not found in devices. Will not be expanded.")

    # Setup layer
    layer_args = {
        "parameter_names": parameter_names,
        "solver": cp.MOSEK,
        "solver_kwargs": {"verbose": False, "accept_unknown": True},
        "add_ground": False,
    }
    layer = zap.DispatchLayer(net, devices, time_horizon=time_horizon, **layer_args)

    # Build objective
    def make_objective(devs):
        f_cost = cost_weight * zap.planning.DispatchCostObjective(net, devs)
        f_emissions = emissions_weight * zap.planning.EmissionsObjective(devs)
        return f_cost + f_emissions

    op_objective = make_objective(devices)
    inv_objective = zap.planning.InvestmentObjective(devices, layer)

    # Setup planning problem
    problem_args = {"lower_bounds": None, "upper_bounds": None, "regularize": regularize}
    problem = zap.planning.PlanningProblem(
        operation_objective=op_objective,
        investment_objective=inv_objective,
        layer=deepcopy(layer),
        **problem_args,
    )

    # Setup stochastic problem
    if stochastic:
        assert hours_per_scenario >= 1
        assert time_horizon % hours_per_scenario == 0

        scenarios = [
            range(i, i + hours_per_scenario) for i in range(0, time_horizon, hours_per_scenario)
        ]
        sub_devices = [[d.sample_time(s, time_horizon) for d in devices] for s in scenarios]

        sub_layers = [
            zap.DispatchLayer(
                devices=d, network=layer.network, time_horizon=hours_per_scenario, **layer_args
            )
            for d in sub_devices
        ]

        sub_op_objectives = [make_objective(d) for d in sub_devices]
        sub_inv_objectives = [
            zap.planning.InvestmentObjective(d, lay) for d, lay in zip(sub_devices, sub_layers)
        ]

        sub_problems = [
            zap.planning.PlanningProblem(
                operation_objective=o,
                investment_objective=i,
                layer=lay,
                **problem_args,
            )
            for o, i, lay in zip(sub_op_objectives, sub_inv_objectives, sub_layers)
        ]

        stochastic_problem = zap.planning.StochasticPlanningProblem(sub_problems)

    else:
        stochastic_problem = None
        sub_devices = None

    return {
        "problem": problem,
        "layer": layer,
        "stochastic_problem": stochastic_problem,
        "sub_devices": sub_devices,
    }


def solve_relaxed_problem(problem, *, should_solve=True, price_bound=50.0, inf_value=50.0):
    problem = problem["problem"]

    if not should_solve:
        print("Skipping relaxation...")
        return None

    else:
        print("Solving relaxation...")
        relaxation = zap.planning.RelaxedPlanningProblem(
            problem,
            max_price=price_bound,
            inf_value=inf_value,
            sd_tolerance=1e-3,
        )

        relaxed_parameters, data = relaxation.solve()
        print(f"Solved relaxation in {data['problem'].solver_stats.solve_time / 60:.2f} minutes.")

        return {
            "relaxation": relaxation,
            "relaxed_parameters": relaxed_parameters,
            "data": data,
            "lower_bound": data["problem"].value,
        }


def solve_problem(
    problem_data,
    relaxation,
    config,
    *,
    use_wandb=False,
    log_wandb_every=2,
    name="gradient_descent",
    initial_state="relaxation",
    num_iterations=10,
    args={"step_size": 1e-3, "num_iterations": 100},
    checkpoint_every=100_000,
    parallel=False,
    ray_num_cpus=16,  # TODO - Deprecated
):
    print("Solving problem...")

    problem: zap.planning.PlanningProblem = problem_data["problem"]
    if problem_data["stochastic_problem"] is not None:
        print("Solving stochastic problem.")
        problem = problem_data["stochastic_problem"]

        if parallel:
            # Setup ray
            print("Initializing Ray.")
            ray_num_cpus = len(problem.subproblems)
            ray.init(num_cpus=ray_num_cpus, _memory=config["system"]["threads"] * (1024**3))
            print(ray.cluster_resources())
            print(ray.cluster_resources()["CPU"])
            print(ray.cluster_resources()["memory"] / 1e9, "GB\n")

            # Remoteize problem
            problem.subproblems = [
                zap.planning.RemotePlanningProblem.remote(p) for p in problem.subproblems
            ]

    # Construct algorithm
    alg = ALGORITHMS[name](**args)

    # Setup wandb
    if use_wandb:
        wandb.init(project="zap", config=config)
        logger = wandb
    else:
        logger = None

    # Initialize
    init = initial_state

    if relaxation is not None and init == "relaxation":
        print("Initializing with relaxation solution.")
        initial_state = deepcopy(relaxation["relaxed_parameters"])

    elif init == "initial":
        print("Initializing with initial parameters (no investment).")
        initial_state = None

    else:
        print("Initializing with a previous solution.")

        # Check if file exists
        initial_path = datadir("results", init, "optimized.json")
        if not initial_path.exists():
            raise ValueError("Could not find initial parameters.")

        with open(initial_path, "r") as f:
            initial_state = json.load(f)

        ref_shapes = {k: v.shape for k, v in problem.initialize_parameters().items()}
        initial_state = {k: np.array(v).reshape(ref_shapes[k]) for k, v in initial_state.items()}

    # Solve
    parameters, history = problem.solve(
        num_iterations=num_iterations,
        algorithm=alg,
        trackers=tr.DEFAULT_TRACKERS,
        initial_state=initial_state,
        wandb=logger,
        log_wandb_every=log_wandb_every,
        lower_bound=relaxation["lower_bound"] if relaxation is not None else None,
        extra_wandb_trackers=get_wandb_trackers(problem_data, relaxation, config),
        checkpoint_every=checkpoint_every,
        checkpoint_func=lambda *args: checkpoint_model(*args, config),
    )

    ray.shutdown()
    if use_wandb:
        wandb.finish()

    return {
        "initial_state": initial_state,
        "parameters": parameters,
        "history": history,
    }


def save_results(relaxation, results, config):
    # Pick a file name
    results_path = get_results_path(config["id"], config.get("index", None))
    results_path.mkdir(parents=True, exist_ok=True)

    # Save relaxation parameters
    if relaxation is not None:
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
    data = load_dataset(**config["data"])
    problem = setup_problem(**data, **config["problem"])

    # Solve relaxation and original problem
    relaxation = solve_relaxed_problem(problem, **config["relaxation"])
    results = solve_problem(problem, relaxation, config, **config["optimizer"])

    save_results(relaxation, results, config)

    return None


def get_wandb_trackers(problem_data, relaxation, config):
    problem, layer = problem_data["problem"], problem_data["layer"]
    is_stochastic = problem_data["stochastic_problem"] is not None
    sub_devices = problem_data["sub_devices"]

    # TODO - Generalize for multi-objective problems
    if is_stochastic:
        carbon_objective = [zap.planning.EmissionsObjective(d) for d in sub_devices]
        cost_objective = [zap.planning.DispatchCostObjective(layer.network, d) for d in sub_devices]
    else:
        carbon_objective = zap.planning.EmissionsObjective(layer.devices)
        cost_objective = zap.planning.DispatchCostObjective(layer.network, layer.devices)

    if is_stochastic:

        def emissions_tracker(J, grad, params, last_state, problem):
            if problem.has_remote_subproblems():
                states = [sub.get_state.remote() for sub in problem.subproblems]
                states = ray.get(states)
            else:
                states = [sub.state for sub in problem.subproblems]

            carbon_costs = [
                c(s, parameters=layer.setup_parameters(**params))
                for c, s in zip(carbon_objective, states)
            ]

            return sum(carbon_costs)

        def cost_tracker(J, grad, params, last_state, problem):
            if problem.has_remote_subproblems():
                states = [sub.get_state.remote() for sub in problem.subproblems]
                states = ray.get(states)
            else:
                states = [sub.state for sub in problem.subproblems]

            fuel_costs = [
                c(s, parameters=layer.setup_parameters(**params))
                for c, s in zip(cost_objective, states)
            ]

            return sum(fuel_costs)

    else:

        def emissions_tracker(J, grad, state, last_state, problem):
            return carbon_objective(problem.state, parameters=layer.setup_parameters(**state))

        def cost_tracker(J, grad, state, last_state, problem):
            return cost_objective(problem.state, parameters=layer.setup_parameters(**state))

    lower_bound = relaxation["lower_bound"] if relaxation is not None else 1.0
    true_relax_cost = (
        problem(**relaxation["relaxed_parameters"]) if relaxation is not None else np.inf
    )
    relax_solve_time = (
        relaxation["data"]["problem"].solver_stats.solve_time if relaxation is not None else 0.0
    )

    # Trackers
    return {
        "emissions": emissions_tracker,
        "fuel_costs": cost_tracker,
        "inv_cost": lambda J, grad, params, last_state, problem: problem.inv_cost.item(),
        "op_cost": lambda J, grad, params, last_state, problem: problem.op_cost.item(),
        "lower_bound": lambda *args: lower_bound,
        "true_relaxation_cost": lambda *args: true_relax_cost,
        "relaxation_solve_time": lambda *args: relax_solve_time,
    }


# ====
# Data Processing Utilities
# ====


def device_index(devices, kind):
    if not any(isinstance(d, kind) for d in devices):
        return None

    return next(i for i, d in enumerate(devices) if isinstance(d, kind))


def get_total_load_curve(devices, every=1, reducer=np.sum):
    devs = [d for d in devices if isinstance(d, zap.Load)]
    total_hourly_load = sum([d.load for d in devs])

    return [
        reducer(total_hourly_load[:, t : t + every])
        for t in range(0, total_hourly_load.shape[1], every)
    ]


# ====
# File I/O and Config Utilities
# ====


def checkpoint_model(parameters, history, config):
    result_dir = get_results_path(config["id"], config.get("index", None))
    result_dir.mkdir(parents=True, exist_ok=True)

    iteration = len(history[tr.LOSS])

    # Save current model
    with open(result_dir / f"model_{iteration:05d}.json", "w") as f:
        ps = {k: v.ravel().tolist() for k, v in parameters.items()}
        json.dump(ps, f)

    return None


def datadir(*args):
    return Path(DATA_PATH, *args)


def projectdir(*args):
    return Path(ZAP_PATH, *args)


def get_results_path(config_name, index=None):
    if index is None:
        return datadir("results", config_name)
    else:
        return datadir("results", config_name, f"{index:03d}")


def parse_scientific_notation(s):
    if isinstance(s, list):
        return [parse_scientific_notation(v) for v in s]
    if isinstance(s, dict):
        return {k: parse_scientific_notation(v) for k, v in s.items()}
    if isinstance(s, str):
        try:
            return float(s)
        except ValueError:
            return s
    return s


# ====
# MAIN FUNCTION
# ====


if __name__ == "__main__":
    config_path = sys.argv[1]

    if len(sys.argv) > 2:
        config_num = int(sys.argv[2])
    else:
        config_num = 0

    config = expand_config(load_config(config_path))[config_num]

    run_experiment(config)
