import cvxpy as cp
from typing import Any

from zap.network import PowerNetwork
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
    ):
        self.network = network
        self.devices = devices
        self.parameter_names = parameter_names
        self.time_horizon = time_horizon
        self.solver = solver

        # TODO - check that parameters match devices
        # TODO - check that parameters are unique?
        pass

    def __call__(self, **kwargs) -> Any:
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        # Check that arguments match parameters
        assert kwargs.keys() == self.parameter_names.keys()

        # Match parameters to devices
        parameters = [{} for _ in self.devices]
        for k, (i, name) in self.parameter_names.items():
            parameters[i][name] = kwargs[k]

        return self.network.dispatch(
            self.devices, time_horizon=self.time_horizon, parameters=parameters, solver=self.solver
        )
