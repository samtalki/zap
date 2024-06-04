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
    ):
        self.network = network
        self.devices = devices
        self.parameter_names = parameter_names
        self.time_horizon = time_horizon
        self.solver = solver

        # TODO - check that parameters match devices
        # TODO - check that parameters are unique?
        pass

    def forward(self, **kwargs) -> ADMMState:
        parameters = self.setup_parameters(**kwargs)

        state, history = self.solver.solve(
            self.network, self.devices, self.time_horizon, parameters=parameters
        )
        self.history = history
        return state

    def backward(self, z, dz, **kwargs):
        # parameters = self.setup_parameters(**kwargs)

        # # dtheta = -JK_theta.T @ inv(JK_z.T) @ dz
        # # dz_bar = inv(JK_z.T) @ dz
        # # start = time.time()
        # dz_bar = self.network.kkt_vjp_variables(
        #     dz, self.devices, z, parameters=parameters, regularize=regularize, vectorize=False
        # )
        # # print("J_var.T @ x: ", time.time() - start)

        # # dtheta = -JK_theta.T @ dz_bar
        # # start = time.time()
        # dtheta = {}
        # for key, (i, name) in self.parameter_names.items():
        #     dtheta[key] = -self.network.kkt_vjp_parameters(
        #         dz_bar,
        #         self.devices,
        #         z,
        #         parameters=parameters,
        #         param_ind=i,
        #         param_name=name,
        #     ).numpy()
        # # print("J_theta.T @ x: ", time.time() - start)

        # return dtheta
        assert NotImplementedError
