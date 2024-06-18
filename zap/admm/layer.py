import numpy as np

from zap.admm.basic_solver import ADMMSolver, ADMMState
from zap.network import PowerNetwork
from zap.devices.abstract import AbstractDevice
from zap.layer import DispatchLayer


class ADMMLayer(DispatchLayer):
    """Maps device parameters to dispatch outcomes."""

    def __init__(
        self,
        network: PowerNetwork,
        devices: list[AbstractDevice],
        parameter_names: dict[str, tuple[int, str]],
        time_horizon: int = 1,
        solver: ADMMSolver = ADMMSolver(num_iterations=100, rho_power=1.0),
        warm_start: bool = True,
        adapt_rho: bool = False,
        adapt_rho_rate: float = 0.1,
    ):
        self.network = network
        self.devices = devices
        self.parameter_names = parameter_names
        self.time_horizon = time_horizon
        self.solver = solver
        self.warm_start = warm_start
        self.adapt_rho = adapt_rho
        self.adapt_rho_rate = adapt_rho_rate

    def forward(self, initial_state=None, **kwargs) -> ADMMState:
        parameters = self.setup_parameters(**kwargs)

        if self.warm_start and initial_state is None and hasattr(self, "state"):
            initial_state = self.state.copy()
        else:
            initial_state = initial_state

        state, history = self.solver.solve(
            self.network,
            self.devices,
            self.time_horizon,
            parameters=parameters,
            initial_state=initial_state,
        )

        self.history = history
        self.state = state

        if self.adapt_rho:
            Jstar, n = history.objective[-1], self.solver.total_terminals
            self.solver.rho_power = self.adapt_rho_rate * Jstar / np.sqrt(n)
            print(f"Reset rho to {self.solver.rho_power}")

        return state

    def backward(self, z, dz, **kwargs):
        assert NotImplementedError
