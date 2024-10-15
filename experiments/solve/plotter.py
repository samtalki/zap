import pickle
import matplotlib.pyplot as plt
import seaborn
from experiments.solve import runner
from pathlib import Path


FIGWIDTH_SMALL = 4.68504  # inches, 119 mm
FIGWIDTH_FULL = 6.5  # inches


def set_full_style():
    seaborn.set_theme(
        style="whitegrid",
        palette="bright",
        rc={
            "axes.edgecolor": "0.15",
            "axes.linewidth": 1.25,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
    )


def set_small_style():
    seaborn.set_theme(
        style="whitegrid",
        palette="bright",
        rc={
            "axes.edgecolor": "0.15",
            "axes.linewidth": 1.25,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )


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
