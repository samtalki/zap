import marimo

__generated_with = "0.7.1"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import pandas as pd
    import datetime as dt

    import torch
    import importlib
    import pypsa
    import json

    from copy import deepcopy
    return cp, deepcopy, dt, importlib, json, mo, np, pd, pypsa, torch


@app.cell
def __():
    import zap
    return zap,


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn
    return plt, seaborn


@app.cell
def __(importlib):
    from experiments import runner

    _ = importlib.reload(runner)
    return runner,


@app.cell
def __(runner):
    _config_list = runner.expand_config(
        runner.load_config("experiments/config/year_compare_v02.yaml")
    )
    config = _config_list[1]
    return config,


@app.cell
def __(config, runner):
    _data = runner.load_dataset(**config["data"])
    net, devices = _data["net"], _data["devices"]
    return devices, net


@app.cell
def __():
    model_id = 300
    return model_id,


@app.cell
def __(json, model_id, np):
    with open(
        f"./data/results/year_compare_v02/000/model_{model_id:05d}.json", "r"
        # f"./data/results/warm_v06/020/model_{model_id:05d}.json", "r"
    ) as f:
        model_state = json.load(f)

    model_state = {
        k: np.array(theta).reshape(-1, 1) for k, theta in model_state.items()
    }
    return f, model_state


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Extract PyPSA data / plot network")
    return


@app.cell
def __(config, model_state, np, pypsa, runner):
    _args = config["data"]["args"]
    _csv_dir = f"{config['data']['case']}/elec_s_{config['data']['num_nodes']}"
    if config["data"]["use_extra_components"]:
        _csv_dir += "_ec"

    pn = pypsa.Network()
    pn.import_from_csv_folder(runner.DATA_PATH / "pypsa/western/" / _csv_dir)

    # Replace data
    pn.generators["p_nom_opt"] = model_state["generator"].ravel()
    pn.lines = pn.lines[~np.isinf(pn.lines.x)]
    pn.lines["p_nom_opt"] = model_state["ac_line"].ravel()
    return pn,


@app.cell
def __(pn):
    gen_colors = pn.carriers.color.to_dict()

    gc_names = list(gen_colors.keys())[:-4]
    gc_vals = list(gen_colors.values())[:-4]
    return gc_names, gc_vals, gen_colors


@app.cell
def __(gc_names):
    gc_names
    return


@app.cell
def __():
    from pathlib import Path
    return Path,


@app.cell
def __(devices):
    sum(d.num_devices for d in devices)
    return


@app.cell
def __(Path, gc_names, gc_vals, plt, pn, pypsa):
    # plt.figure(figsize=()

    _gen = (
        pn.generators.assign(pnom=pn.generators.p_nom_opt)
        .groupby(["bus", "carrier"])
        .p_nom_opt.sum()
    )
    artists = pn.plot(
        bus_sizes=_gen / 30.0,
        bus_colors=pn.carriers.color.to_dict(),
        # margin=0.5,
        # flow="mean",
        line_widths=0.25 * pn.lines.p_nom_opt,
        line_colors="black",
        # link_widths=0,
    )
    pypsa.plot.add_legend_patches(
        plt.gca(), gc_vals, gc_names, legend_kw={"fontsize": 10, "loc": "upper right", "bbox_to_anchor": (1.5, 1.0)}
    )
    plt.tight_layout()
    plt.savefig(Path().home() / "figures/gpu_perf/network.svg")

    artists
    return artists,


if __name__ == "__main__":
    app.run()
