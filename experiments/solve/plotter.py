import pickle
import zap

from experiments.solve import runner

from pathlib import Path


def open_solver_data(config):
    path = Path(runner.get_results_path(config["id"], config["index"]))
    with open(path / "solver_data.pkl", "rb") as f:
        solver_data = pickle.load(f)

    return solver_data


def open_solution(config):
    path = Path(runner.get_results_path(config["id"], config["index"]))
    with open(path / "ys.pkl", "rb") as f:
        solution = pickle.load(f)

    return solution


def open_cases(config):
    path = Path(runner.get_results_path(config["id"], config["index"]))
    with open(path / "cases.pkl", "rb") as f:
        cases = pickle.load(f)

    return cases


def reverse_index(case_index, arg_index, num_cases):
    # Index a list of length num_cases*num_args by case_index and arg_index
    # where arg_index is the outer index and case_index is the inner index
    return arg_index * num_cases + case_index
