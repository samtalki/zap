import sys
import pypsa
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path

import zap


ZAP_PATH = Path(zap.__file__).parent.parent
DATA_PATH = ZAP_PATH / "data"


def load_config(config_path):
    # TODO Expand configs
    # TODO Hunt for improperly parsed scientific notation
    pass


def load_dataset(config):
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

        if not data["use_batteries"]:
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
    pass


def solve_relaxed_problem(data, problem, config):
    pass


def solve_problem(data, problem, relaxation, config):
    pass


def save_results(relaxation, results, config):
    pass


def run_experiment(config):
    # Load data and formulate problem
    data = load_dataset(config)
    problem = setup_problem(data, config)

    # Solve relaxation and original problem
    relaxation = solve_relaxed_problem(data, problem, config)
    results = solve_problem(data, problem, relaxation, config)

    save_results(relaxation, results, config)


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    run_experiment(config)
