import marimo

__generated_with = "0.7.1"
app = marimo.App()


@app.cell
def __():
    import time
    import pickle

    import marimo as mo

    from copy import deepcopy
    from pathlib import Path
    return Path, deepcopy, mo, pickle, time


@app.cell
def __():
    import zap
    import runner
    return runner, zap


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn

    SEABORN_RC = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }

    seaborn.set_theme(style="whitegrid", rc=SEABORN_RC)
    return SEABORN_RC, plt, seaborn


@app.cell
def __(Path, runner):
    CONFIG_NAME = "gpu_exp_v01.yaml"
    CONFIG = runner.load_config("experiments/config/" + CONFIG_NAME)
    RESULTS_DIR = Path("data/gpu_perf/").absolute()

    DEFAULT_NODE = 500
    DEFAULT_HOUR = 96

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    return CONFIG, CONFIG_NAME, DEFAULT_HOUR, DEFAULT_NODE, RESULTS_DIR


@app.cell(hide_code=True)
def __(DEFAULT_HOUR, DEFAULT_NODE):
    def get_name(
        num_nodes=DEFAULT_NODE,
        num_hours=DEFAULT_HOUR,
    ):
        return f"case_{num_nodes}_{num_hours}"
    return get_name,


@app.cell(hide_code=True)
def __(CONFIG, DEFAULT_HOUR, DEFAULT_NODE, deepcopy, runner):
    def setup_case(
        num_nodes=DEFAULT_NODE,
        num_hours=DEFAULT_HOUR,
        scale_load=1.0,
    ):
        config = deepcopy(CONFIG)
        config["data"]["num_nodes"] = num_nodes
        config["data"]["num_hours"] = num_hours
        config["data"]["args"]["scale_load"] = scale_load

        # Load network
        dataset = runner.load_dataset(**config["data"])
        net, devices = dataset["net"], dataset["devices"]

        return config, net, devices, f"case_{num_nodes}_{num_hours}"
    return setup_case,


@app.cell(hide_code=True)
def __(RESULTS_DIR, deepcopy, pickle, runner, time):
    def run_case_cvx(config, net, devices, name):
        config = deepcopy(config)

        # Formulate problem
        problem_data = runner.setup_problem(
            net,
            devices,
            use_admm=False,
            **config["problem"],
        )
        layer = problem_data["layer"]
        theta = layer.initialize_parameters(None)

        # Solve problem
        t0 = time.perf_counter()
        y = layer(**theta)
        t1 = time.perf_counter()

        data = {
            "state": y,
            "layer": layer,
            "run_time": t1 - t0,
            "solve_time": y.problem.solver_stats.solve_time,
        }
        pickle.dump(data, open(RESULTS_DIR / f"{name}_cvx.pkl", "wb"))
        print(f"Mosek solved case in {t1 - t0} seconds.")
        return data
    return run_case_cvx,


@app.cell(hide_code=True)
def __(RESULTS_DIR, deepcopy, pickle, runner, time):
    def run_case_admm(config, net, devices, name):
        config = deepcopy(config)

        # Formulate problem
        problem_data = runner.setup_problem(
            net, devices, **config["problem"], **config["layer"]
        )
        layer = problem_data["layer"]
        theta = layer.initialize_parameters(None)

        # Solve problem
        t0 = time.perf_counter()
        y = layer(**theta)
        t1 = time.perf_counter()

        data = {
            "state": y,
            "layer": layer,
            "run_time": t1 - t0,
            "solve_time": t1 - t0,
            "history": deepcopy(layer.history),
        }
        pickle.dump(data, open(RESULTS_DIR / f"{name}_admm.pkl", "wb"))
        print(f"ADMM solved case in {t1 - t0} seconds.")
        return data
    return run_case_admm,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Experiment 1 - Scaling")
    return


@app.cell(hide_code=True)
def __(DEFAULT_HOUR, RESULTS_DIR, get_name, pickle):
    def load_data(nodes, hours=[DEFAULT_HOUR]):
        data_cvx = []
        data_admm = []

        for _n in nodes:
            for h in hours:
                data_cvx += [
                    pickle.load(
                        open(RESULTS_DIR / f"{get_name(_n, h)}_cvx.pkl", "rb")
                    )
                ]
                data_admm += [
                    pickle.load(
                        open(RESULTS_DIR / f"{get_name(_n, h)}_admm.pkl", "rb")
                    )
                ]

        return data_cvx, data_admm
    return load_data,


@app.cell
def __(plt):
    def plot_scaling(x, data_cvx, data_admm, xlabel="", logscale=False):
        fig, ax = plt.subplots(figsize=(3, 2))

        cvx_times = [d["run_time"] for d in data_cvx]
        admm_times = [d["run_time"] for d in data_admm]

        ax.plot(x, cvx_times, label="Mosek", marker="o", ms=4)
        ax.plot(x, admm_times, label="Message Passing", marker="o", ms=4)

        ax.legend()
        # ax.set_ylabel("Time [seconds]")
        # ax.set_xlabel(xlabel)
        if logscale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, ax.get_ylim()[1])

        return fig, ax
    return plot_scaling,


@app.cell(hide_code=True)
def __(RESULTS_DIR, get_name, run_case_admm, run_case_cvx, setup_case):
    def gen_data(nodes, hours, replace=False):
        # Generate data
        for _h in hours:
            for _n in nodes:
                cvx_path = RESULTS_DIR / f"{get_name(_n, _h)}_cvx.pkl"
                admm_path = RESULTS_DIR / f"{get_name(_n, _h)}_admm.pkl"

                if replace or not cvx_path.exists():
                    _case = setup_case(num_nodes=_n, num_hours=_h)
                    run_case_cvx(*_case)

                if replace or not admm_path.exists():
                    _case = setup_case(num_nodes=_n, num_hours=_h)
                    run_case_admm(*_case)
    return gen_data,


@app.cell(hide_code=True)
def __(mo):
    mo.md("### 1A - Nodes")
    return


@app.cell
def __():
    EXP_A_NODES = [100, 200, 500, 1000]
    EXP_A_HOURS = [24]
    return EXP_A_HOURS, EXP_A_NODES


@app.cell
def __(EXP_A_NODES, Path, load_data, plot_scaling):
    _fig, _ax = plot_scaling(
        EXP_A_NODES,
        *load_data(EXP_A_NODES, hours=[24]),
        xlabel="Network Size",
        logscale=True
    )
    _ax.set_ylim(1e-1, 1e2)
    _ax.legend(loc="upper center")
    _fig.tight_layout()
    _fig.savefig(Path().home() / "figures/gpu_perf/scaling_24.svg")
    _fig
    return


@app.cell
def __(EXP_A_NODES, Path, load_data, plot_scaling):
    _fig, _ax = plot_scaling(
        EXP_A_NODES,
        *load_data(EXP_A_NODES, hours=[96]),
        xlabel="Network Size",
        logscale=True
    )
    _ax.set_ylim(1e-1, 1e2)
    _ax.get_legend().remove()
    _fig.tight_layout()
    _fig.savefig(Path().home() / "figures/gpu_perf/scaling_96.svg")
    _fig
    return


@app.cell
def __(EXP_A_HOURS, EXP_A_NODES, gen_data):
    gen_data(EXP_A_NODES, EXP_A_HOURS, replace=False)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("### 1B - Hours")
    return


@app.cell
def __():
    EXP_B_NODES = [500]
    EXP_B_HOURS = [24, 48, 96, 192, 384]
    return EXP_B_HOURS, EXP_B_NODES


@app.cell
def __(EXP_B_HOURS, EXP_B_NODES, load_data, plot_scaling):
    _fig, _ax = plot_scaling(
        EXP_B_HOURS,
        *load_data(EXP_B_NODES, hours=EXP_B_HOURS),
        xlabel="Time Horizon",
        logscale=True,
    )

    _fig
    return


@app.cell
def __(EXP_B_HOURS, EXP_B_NODES, gen_data):
    gen_data(EXP_B_NODES, EXP_B_HOURS, replace=False)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"## Experiment 2 - Time for Backwards Pass")
    return


if __name__ == "__main__":
    app.run()
