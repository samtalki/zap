import marimo

__generated_with = "0.4.3"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import torch
    return mo, torch


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Vectorized Operations")
    return


@app.cell
def __():
    num_devices = 2000
    num_timesteps = 256 * 24
    return num_devices, num_timesteps


@app.cell
def __(num_devices, num_timesteps, torch):
    A0 = torch.rand(num_devices, num_timesteps)
    A1 = torch.rand(num_devices, num_timesteps)
    print(A0.device, A1.device)
    return A0, A1


@app.cell
def __(A0, A1):
    B0 = A0.to(device="cuda")
    B1 = A1.to(device="cuda")
    print(B0.device, B1.device)
    return B0, B1


@app.cell
def __():
    num_tests = 100
    return num_tests,


@app.cell
def __(A0, A1, num_tests):
    for _ in range(num_tests):
        _A2 = A0.numpy() * A1.numpy()
    return


@app.cell
def __(A0, A1, num_tests):
    for _ in range(num_tests):
        _A2 = A0 * A1
    return


@app.cell
def __(B0, B1, num_tests):
    for _ in range(num_tests):
        _B2 = B0 * B1
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Sparse Operations")
    return


@app.cell
def __():
    # TODO - try scatter add
    return


if __name__ == "__main__":
    app.run()
