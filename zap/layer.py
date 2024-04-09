import cvxpy as cp
from typing import Any

from zap.network import PowerNetwork, DispatchOutcome
from zap.devices.abstract import AbstractDevice


class DispatchLayer:
    """Maps device parameters to dispatch outcomes."""

    def __init__(
        self,
        network: PowerNetwork,
        devices: list[AbstractDevice],
        parameter_names: dict[str, tuple[int, str]],
        time_horizon: int = 1,
        solver=cp.ECOS,
        solver_kwargs={},
    ):
        self.network = network
        self.devices = devices
        self.parameter_names = parameter_names
        self.time_horizon = time_horizon
        self.solver = solver
        self.solver_kwargs = solver_kwargs

        # TODO - check that parameters match devices
        # TODO - check that parameters are unique?
        pass

    def __call__(self, **kwargs) -> Any:
        return self.forward(**kwargs)

    def setup_parameters(self, **kwargs):
        # Check that arguments match parameters
        assert kwargs.keys() == self.parameter_names.keys()

        # Match parameters to devices
        parameters = [{} for _ in self.devices]
        for k, (i, name) in self.parameter_names.items():
            parameters[i][name] = kwargs[k]

        return parameters

    def forward(self, **kwargs) -> DispatchOutcome:
        parameters = self.setup_parameters(**kwargs)

        return self.network.dispatch(
            self.devices,
            time_horizon=self.time_horizon,
            parameters=parameters,
            solver=self.solver,
            solver_kwargs=self.solver_kwargs,
        )

    def backward(self, z: DispatchOutcome, dz: DispatchOutcome, regularize=1e-8, **kwargs):
        parameters = self.setup_parameters(**kwargs)

        # dtheta = -JK_theta.T @ inv(JK_z.T) @ dz
        # dz_bar = inv(JK_z.T) @ dz
        # start = time.time()
        dz_bar = self.network.kkt_vjp_variables(
            dz, self.devices, z, parameters=parameters, regularize=regularize, vectorize=False
        )
        # print("J_var.T @ x: ", time.time() - start)

        # dtheta = -JK_theta.T @ dz_bar
        # start = time.time()
        dtheta = {}
        for key, (i, name) in self.parameter_names.items():
            dtheta[key] = -self.network.kkt_vjp_parameters(
                dz_bar,
                self.devices,
                z,
                parameters=parameters,
                param_ind=i,
                param_name=name,
            ).numpy()
        # print("J_theta.T @ x: ", time.time() - start)

        return dtheta
