import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 1 - ⚡ Zap Basics ⚡


        In this notebook, we introduce the basic types and functionality in Zap.
        We will manually construct a small electricity network, solve a dispatch problem using both CVXPY and ADMM, and analyze the results.

        1. Creating a Network
        2. Solving Dispatch Problems
        3. Analyzing Results
        4. Sensitivities and Differentiable Programming
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Creating a Network

        Networks in Zap consist of a `Network` object, which defines the underlying network, and a list of `Device` objects, which are electrical components attached to the network. Devices include generators, loads, transmission lines, batteries, and all other components that connect to the network. Devices can also be used to model financial contracts, such as bids and offers for energy or transmission congestion contracts.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Constructing Devices""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""First, let's initialize a simple network with 3 nodes.""")
    return


@app.cell
def __():
    import zap
    import numpy as np
    return np, zap


@app.cell
def __(zap):
    net = zap.PowerNetwork(num_nodes=3)
    return (net,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Now, let's add a load to the network.""")
    return


@app.cell
def __(net, np, zap):
    load = zap.Load(
        num_nodes=net.num_nodes,
        # The terminal of the device is the node to which it connects
        terminal=np.array([0]),
        # The load argument is a time series of the power consumed by the device
        # This is 2-d array, see more below!
        load=np.array([[10.0, 15.0, 20.0]]),
        # This is the curtailment cost of the device, which we will assume is $500 / MWh
        linear_cost=np.array([500.0]),
    )
    return (load,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        All data passed to the device must be `numpy` arrays or of a similar type (e.g., a PyTorch `Tensor`).
        We can also create groups of devices at the same time.
        Let's add two generators to the network.
        """
    )
    return


@app.cell
def __(net, np, zap):
    generators = zap.Generator(
        num_nodes=net.num_nodes,
        # Since we have two generators, we specify two terminals
        terminal=np.array([1, 2]),
        # Nominal capacity refers to the generator nameplate capacity
        nominal_capacity=np.array([50.0, 20.0]),
        # This is the marginal cost of generation
        linear_cost=np.array([0.1, 30.0]),
        # Emissions rates are optional, but useful for many applications
        emission_rates=np.array([0.0, 500.0]),
        # Dynamic capacity refers to the time-varying capacity factor of each generator
        dynamic_capacity=np.array(
            [
                [0.1, 0.5, 0.1],
                [1.0, 1.0, 1.0],
            ]
        ),
    )
    return (generators,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Since we have two generators, we specify both their parameters at the same time.
        Some of the parameters, such as costs or dynamic capacities, can also time varying, in which case the user must pass a 2d-array of data.
        The first dimension of the data specifies which device is being referred to, and the second dimension of the data specifies the time period. 
        If you pass a 1d-array, Zap will automatically assume this is a static quantity that does not change over time.

        Finally, let's initialize a few transmission lines to connect our network.
        """
    )
    return


@app.cell
def __(net, np, zap):
    lines = zap.ACLine(
        num_nodes=net.num_nodes,
        # AC lines are two-terminal devices, so we specify both a source and sink terminal for each line
        # We will build the classic 3-bus "triangle" network
        source_terminal=np.array([0, 1, 2]),
        sink_terminal=np.array([1, 2, 0]),
        nominal_capacity=np.ones(3),
        # This is the per-MW susceptance, so the susceptance of each line is its nominal capacity times its susceptance
        susceptance=np.array([1.0, 1.0, 1.0]),
        # Lines can also have time-vary capacities to simulate outages or de-ratings
        # In this example, we will just make them static
        capacity=np.ones(3),
    )
    return (lines,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Putting the Network Together

        To finish up, we will create a list of device (groups). 
        Together with the network and the time horizon, this list fully specifies an electrical network!
        """
    )
    return


@app.cell
def __(generators, lines, load, net):
    devices = [load, generators, lines]
    time_horizon = 3

    net, devices, time_horizon  # Fully network specification
    return devices, time_horizon


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        Before we move on, here are a couple of notes about constructing devices.

        - Many arguments (capital cost, emissions rates, line slacks, etc.) are optional and introduce additional functionality, e.g., for planning studies.
        - For performance reasons, it's best to specify a few device groups (ideally, one for each type of device) with many devices per group, instead of many groups with one or just a few devices per group. This is so that computations can be efficiently vectorized, both on CPU and GPU machines.
        - Remember that the first dimension (rows) of a parameter specifies which device in the group the parameter is for, and the second dimension (columns) specifies the time period. **Dynamic data is always 2-dimensional!**
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Solving Dispatch Problems""")
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
