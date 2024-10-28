import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import datetime as dt
    import pandas as pd
    import numpy as np
    import scipy.sparse as sp

    import torch
    import pypsa

    return dt, mo, np, pd, pypsa, sp, torch


@app.cell
def __():
    import zap

    return (zap,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Load Network Data""")
    return


@app.cell
def __(pypsa):
    pn = pypsa.Network()
    pn.import_from_csv_folder("data/pypsa/western/load_medium/elec_s_4000")
    return (pn,)


@app.cell
def __():
    time_horizon = 24
    return (time_horizon,)


@app.cell(hide_code=True)
def __(dt, np, pd, pn, time_horizon, zap):
    _start_date = dt.datetime(2019, 1, 2, 0)

    dates = pd.date_range(
        _start_date,
        _start_date + dt.timedelta(hours=time_horizon),
        freq="1h",
        inclusive="left",
    )

    net, devices = zap.importers.load_pypsa_network(
        pn,
        dates,
        power_unit=1000.0,
        cost_unit=100.0,
        # Costs
        marginal_load_value=500.0,
        load_cost_perturbation=10.0,
        generator_cost_perturbation=1.0,
        # Rescale capacities
        scale_load=0.6,
        scale_generator_capacity_factor=0.7,
        scale_line_capacity_factor=0.7,
        # Empty generators
        drop_empty_generators=False,
        expand_empty_generators=0.5,
        # Battery stuff
        battery_discharge_cost=1.0,
        battery_init_soc=0.0,
        battery_final_soc=0.0,
    )

    _ground = zap.Ground(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        voltage=np.array([0.0]),
    )
    devices += [_ground]
    return dates, devices, net


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Transfer to CUDA""")
    return


@app.cell
def __(torch):
    machine, dtype = "cuda", torch.float32
    return dtype, machine


@app.cell
def __(devices, dtype, machine, torch):
    torch.cuda.empty_cache()
    torch_devices = [d.torchify(machine=machine, dtype=dtype) for d in devices]
    return (torch_devices,)


@app.cell
def __(
    contingency_device,
    dtype,
    machine,
    net,
    num_contingencies,
    time_horizon,
    torch_devices,
    torch_mask,
    zap,
):
    admm = zap.admm.ADMMSolver(
        num_iterations=5,
        rho_power=1.0,
        machine=machine,
        dtype=dtype,
    )
    state, history = admm.solve(
        net,
        torch_devices,
        time_horizon,
        num_contingencies=num_contingencies,
        contingency_device=contingency_device,
        contingency_mask=torch_mask,
    )
    return admm, history, state


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Build Contingency Set""")
    return


@app.cell
def __():
    num_contingencies = 500
    contingency_device = 3
    return contingency_device, num_contingencies


@app.cell
def __(
    contingency_device,
    devices,
    dtype,
    machine,
    num_contingencies,
    sp,
    torch,
):
    contingency_mask = sp.lil_matrix((num_contingencies, devices[contingency_device].num_devices))

    for c in range(num_contingencies):
        contingency_mask[c, c] = 1.0

    contingency_mask = contingency_mask.tocsr()

    torch_mask = torch.tensor(contingency_mask.todense(), device=machine, dtype=dtype)
    torch_mask = torch.vstack(
        [
            torch.zeros(torch_mask.shape[1], device=machine, dtype=dtype),
            torch_mask,
        ]
    )
    return c, contingency_mask, torch_mask


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Benchmark Sparse Matrix Multiplies""")
    return


@app.cell
def __():
    from torch.profiler import profile, ProfilerActivity

    return ProfilerActivity, profile


@app.cell
def __(contingency_device, devices, state):
    dev = devices[contingency_device]
    X = state.power[contingency_device][0]
    X_swapped = X.swapaxes(0, 2).swapaxes(1, 2).clone()
    return X, X_swapped, dev


@app.cell
def __(dev, dtype, machine, torch, zap):
    apply_sparse_A = lambda X: zap.admm.util.apply_incidence(dev, [X])[0]

    A_dense = torch.tensor(dev.incidence_matrix[0].todense(), device=machine, dtype=dtype)
    apply_dense_A = lambda X: A_dense @ X
    return A_dense, apply_dense_A, apply_sparse_A


@app.cell
def __(A_dense, X_swapped):
    print(A_dense.shape)
    print(X_swapped.shape)
    return


@app.cell
def __(ProfilerActivity, X, apply_sparse_A, profile):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof1:
        apply_sparse_A(X)

    print(prof1.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    return (prof1,)


@app.cell
def __(ProfilerActivity, X_swapped, apply_dense_A, profile):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof2:
        apply_dense_A(X_swapped)

    print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    return (prof2,)


if __name__ == "__main__":
    app.run()
