import sys
import zap


def load_config(config_path):
    pass


def load_dataset(config):
    data = config["data"]

    if data["name"] == "pypsa":
        pn = None
        dates = None
        net, devices = zap.importers.load_pypsa_network(pn, dates, **data["args"])

        if data["add_ground"]:
            raise NotImplementedError

        if data["use_batteries"]:
            raise NotImplementedError

    else:
        raise ValueError("Unknown dataset")


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
