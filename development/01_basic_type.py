import marimo

__generated_with = "0.2.8"
app = marimo.App()


@app.cell
def __():
    import torch
    import zap
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import scipy.sparse as sp
    import matplotlib.pyplot as plt

    import importlib
    importlib.reload(zap)
    return cp, importlib, mo, np, plt, sp, torch, zap


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Define Network")
    return


@app.cell
def __():
    num_nodes = 7
    return num_nodes,


@app.cell
def __(num_nodes, zap):
    net = zap.PowerNetwork(num_nodes)
    return net,


@app.cell
def __(np, num_nodes, zap):
    generators = zap.Generator(
        num_nodes=num_nodes,
        terminal=np.array([0, 1, 3]),
        dynamic_capacity=np.array(
            [
                [100.0, 100.0, 100.0, 100.0],  # Peaker
                [10.0, 50.0, 50.0, 15.0],  # Solar panel
                [15.0, 15.0, 15.0, 15.0],  # CC Gas
            ]
        ),
        linear_cost=np.array([100.0, 0.5, 40.0]),
    )

    loads = zap.Load(
        num_nodes=num_nodes,
        terminal=np.array([0]),
        load=np.array([[30.0, 40.0, 45.0, 80.0]]),
        linear_cost=np.array([200.0]),
    )

    links = zap.ACLine(
        num_nodes=num_nodes,
        source_terminal=np.array([0, 1, 3]),
        sink_terminal=np.array([1, 3, 0]),
        capacity=np.array([45.0, 50.0, 11.0]),
        susceptance=np.array([0.1, 0.05, 1.0]),
        linear_cost=0.025 * np.ones(3)
    )

    batteries = zap.Battery(
        num_nodes=num_nodes,
        terminal=np.array([1]),
        power_capacity=np.array([5.0]),
        duration=np.array([4.0]),
        linear_cost=np.array([0.01]),
    )
    return batteries, generators, links, loads


@app.cell
def __(links):
    lt_incidence = [A.todense() for A in links.incidence_matrix]
    lt_incidence[0] - lt_incidence[1]
    return lt_incidence,


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Solve Dispatch Problem")
    return


@app.cell
def __(batteries, generators, links, loads, net, np):
    devices = [generators, loads, links, batteries]
    parameters = [
        {"nominal_capacity": np.array([1.0, 2.0, 1.0])},
        {},
        {"nominal_capacity": np.array([1.0, 1.0, 1.0])},
        {"power_capacity": np.array([5.0])},
    ]
    result = net.dispatch(
        devices,
        time_horizon=4,
        parameters=parameters,
    )

    print(result.problem.status, result.problem.value)
    print("Generator Powers")
    print(result.power[0][0].round(decimals=2))
    # print(result.prices.round(decimals=2))

    print("\nBattery Powers")
    print(result.power[3])
    print("Battery SOC")
    print(result.local_variables[3][0])

    print("Total Solar")
    print(np.sum(result.power[0][0][1, :]))
    return devices, parameters, result


@app.cell
def __(devices, net, np, parameters, result):
    K = net.kkt(devices, result, parameters=parameters)
    K_vec = K.vectorize()

    print(K_vec.shape)
    print(f"KKT Error: {np.linalg.norm(K_vec)}")
    print(1e-6 * np.sqrt(K_vec.size))

    result.vectorize()
    result.size
    result.blocks

    devs = devices + [result.ground]
    _i = 0

    cmat = devs[_i].equality_matrices(
        result.local_equality_duals[_i],
        result.power[_i],
        result.angle[_i],
        result.local_variables[_i],
    )
    cmat
    return K, K_vec, cmat, devs


@app.cell
def __(np, pn_devices, pn_net, pn_result, torch, zap):
    # Torch stuff

    _i = 0
    _key = "nominal_capacity"
    _devices = pn_devices
    _result = pn_result
    _net = pn_net

    _params = [{} for _ in _devices]
    _params[_i][_key] = _devices[_i].nominal_capacity

    _result_tc = _result.torchify(requires_grad=True)
    _grad_tc = _result.package(np.zeros_like(_result.vectorize()))
    _grad_tc.power[_i][0] += _devices[0].linear_cost

    # ====
    # Use network VJP
    # ====

    # grad_line_capacity = _net.kkt_vjp_parameters(
    #     _grad_tc,
    #     _devices,
    #     _result,
    #     parameters=_params,
    #     param_ind=_i,
    #     param_name=_key,
    # )

    # ====


    _tc_params_i = {k: v for k, v in _params[_i].items()}
    _tc_params_i[_key] = zap.util.torchify(_tc_params_i[_key], requires_grad=True)

    # ====
    # Backwards on dev constraints
    # =====

    _constr_ind = 1
    ineq_constrs = _devices[_i].inequality_constraints(
        _result_tc.power[_i],
        _result_tc.angle[_i],
        _result_tc.local_variables[_i],
        **_tc_params_i,
        la=torch,
    )
    ineq_constrs[_constr_ind].backward(
        _result_tc.local_inequality_duals[_i][_constr_ind]
    )
    foo = _tc_params_i[_key].grad

    # ====
    # Backwards on Lagrange grad
    # =====

    # dLdx = _devices[_i].lagrangian_gradients(
    #     _result_tc.power[_i],
    #     _result_tc.angle[_i],
    #     _result_tc.local_variables[_i],
    #     _result_tc.local_equality_duals[_i],
    #     _result_tc.local_inequality_duals[_i],
    #     la=torch,
    #     **_tc_params_i,
    # )


    # _L = _devices[_i].lagrangian(
    #     _result_tc.power[_i],
    #     _result_tc.angle[_i],
    #     _result_tc.local_variables[_i],
    #     _result_tc.local_equality_duals[_i],
    #     _result_tc.local_inequality_duals[_i],
    #     la=torch,
    #     **_tc_params_i,
    # )

    # # Gradient of Lagrangian
    # dLda = torch.autograd.grad(_L, _result_tc.power[_i], create_graph=True)

    # # Parameter Jacobian of gradient of Lagrangian
    # _grad_tc.power[_i][0] += 2.0
    # # _grad_tc.angle[_i][1] += 1.0

    # dLda_dtheta = torch.autograd.grad(
    #     dLda,
    #     _tc_params_i[_key],
    #     grad_outputs=[zap.util.torchify(p) for p in _grad_tc.power[_i]],
    #     allow_unused=True,
    # )

    # ====
    # Output
    # ====

    foo
    return foo, ineq_constrs


@app.cell
def __(plt, pn_devices, pn_net, pn_result):
    # Build Jacobian
    J = pn_net.kkt_jacobian_variables(pn_devices, pn_result)  #, parameters=parameters)

    # Plot
    plt.spy(J, ms=0.1, color="black")
    return J,


@app.cell
def __(J, np, result, sp):
    tol = 1e-4

    np.random.seed(0)
    _y = np.random.randn(J.shape[0])

    # Factorize and solve
    # (1) Regularize and use LU
    J_reg = J.T.tocsc()
    J_reg += tol*sp.eye(J.shape[0])
    J_lu = sp.linalg.splu(J_reg)
    _x1 = J_lu.solve(_y)

    result.package(_x1).power
    return J_lu, J_reg, tol


@app.cell
def __(np, pn_result, result):
    # # Vectorization
    # # TODO - Move to tests
    print(
        np.testing.assert_equal(
            result.package(result.vectorize()).vectorize(), result.vectorize()
        )
    )
    print(
        np.linalg.norm(
            pn_result.package(pn_result.vectorize()).vectorize()
            - pn_result.vectorize()
        )
    )

    # # KKT Jacobian
    # # TODO - Move to tests
    # _x = pn_result
    # _devs = pn_devices
    # _params = None

    # def kkt_jac_test(_delta):
    #     # Create perturbation
    #     _K0 = net.kkt(_devs, _x, parameters=_params)
    #     _dx = np.random.randn(*_K0.vectorize().shape)
    #     _dx = _dx / np.linalg.norm(_dx)
    #     _dx *= _delta

    #     # Compute Jacobian
    #     _JK = net.kkt_jacobian_variables(_devs, _x, parameters=_params)

    #     # Estimated change in KKT after perturbation
    #     _dK_est = _JK @ _dx

    #     # True change in KKT after perturbation
    #     _perturbed_x = _x.package(_x.vectorize() + _dx)

    #     _dK_true = net.kkt(_devs, _perturbed_x, parameters=_params).vectorize()
    #     _dK_true -= _K0.vectorize()

    #     # Measure differences
    #     _delta_dK = _dK_true - _dK_est
    #     _norm_dK = np.linalg.norm(_delta_dK)

    #     print(f"||dK||: {_norm_dK}")
    #     print(f"||dK|| / ||dx||: {_norm_dK / np.linalg.norm(_dx)}")

    #     # Look entrywise
    #     return _x.package(
    #         np.round(_delta_dK / np.linalg.norm(_dx), decimals=8)
    #     )

    # for _delta in np.logspace(0, -4, num=5):
    #     print(f"\n\n||dx|| = {_delta}")
    #     _diffs = kkt_jac_test(_delta)
    return


@app.cell
def __():
    # net.kkt_jacobian_variables(
    #     devices, result, parameters=parameters, vectorize=False
    # ).local_inequality_duals
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Package in Layer")
    return


@app.cell
def __(batteries, generators, links, loads, net, np, result, zap):
    layer = zap.DispatchLayer(
        net,
        [generators, loads, links, batteries],
        parameter_names={"generator_capacity": (0, "nominal_capacity")},
        time_horizon=4,
    )

    capacities = np.array([1.0, 2.0, 1.0])
    result2 = layer(generator_capacity=capacities)

    np.linalg.norm(result2.power[0][0] - result.power[0][0])
    return capacities, layer, result2


@app.cell
def __(np, result, result2):
    np.linalg.norm(result2.vectorize() - result.vectorize())
    np.allclose(result2.vectorize(), result.vectorize())
    return


@app.cell
def __(capacities, layer, np, result2):
    _dz = result2.package(np.zeros_like(result2.vectorize()))
    _dz.power[0][0] += np.array([[0.0, 1.0, 0.0]]).T

    _dtheta = layer.backward(result2, _dz, generator_capacity=capacities)

    _dtheta
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## Import a Network From PyPSA")
    return


@app.cell
def __():
    import pypsa
    import pandas as pd
    import datetime as dt

    pd.set_option("future.no_silent_downcasting", True)
    return dt, pd, pypsa


@app.cell
def __(pypsa):
    pn = pypsa.Network(
        "../Epsilon/.pypsa/workflow/resources/western/elec_s_100.nc"
    )
    return pn,


@app.cell
def __(dt, pd, pn, zap):
    dates = pd.date_range(
        dt.datetime(2019, 1, 2, 0),
        dt.datetime(2019, 1, 2, 0) + dt.timedelta(hours=24),
        freq="1h",
        inclusive="left",
    )

    pn_net, pn_devices = zap.load_pypsa_network(pn, dates)
    return dates, pn_devices, pn_net


@app.cell
def __(cp, dates, pn_devices, pn_net):
    print(
        f"Solving a problem with {pn_net.num_nodes} nodes and {len(dates)} time periods."
    )
    pn_result = pn_net.dispatch(
        pn_devices,
        time_horizon=len(dates),
        solver=cp.MOSEK
    )
    print(f"Solved in {pn_result.problem.solver_stats.solve_time} seconds.")
    return pn_result,


@app.cell
def __(np, pn_devices, pn_result):
    curtailed_load = np.sum(
        pn_result.power[1][0] - pn_devices[1].min_power
    ) / np.sum(pn_devices[1].min_power)
    curtailed_load
    return curtailed_load,


@app.cell
def __(devices, zap):
    [i for i, d in enumerate(devices) if isinstance(d, zap.ACLine)][0]
    return


if __name__ == "__main__":
    app.run()
