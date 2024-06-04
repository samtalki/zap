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
    ):
        self.network = network
        self.devices = devices
        self.parameter_names = parameter_names
        self.time_horizon = time_horizon
        self.solver = solver
        self.warm_start = warm_start

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
        return state

    def backward(self, z, dz, **kwargs):
        assert NotImplementedError
