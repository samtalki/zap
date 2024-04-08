import dataclasses
import numpy as np

from zap.devices.abstract import AbstractDevice
from zap.admm.util import (
    nested_add,
    nested_subtract,
    nested_norm,
    get_num_terminals,
    dc_average,
    ac_average,
    get_terminal_residual,
)


@dataclasses.dataclass
class ADMMState:
    num_terminals: object
    num_ac_terminals: object
    power: object
    phase: object
    dual_power: object
    dual_phase: object
    avg_power: object = None
    avg_phase: object = None
    resid_power: object = None
    resid_phase: object = None
    objective: object = None

    def update(self, **kwargs):
        """Return a new state with fields updated."""
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass
class ADMMSolver:
    """Stores ADMM solver parameters and exposes a solve function."""

    num_iterations: int
    rho_power: float
    rho_angle: float = None
    atol: float = 0.0
    rtol: float = 1e-3
    resid_norm: object = None
    safe_mode: bool = False
    track_objective: bool = True

    def get_rho(self):
        rho_power = self.rho_power
        rho_angle = self.rho_angle

        if rho_angle is None:
            rho_angle = rho_power

        return rho_power, rho_angle

    def solve(
        self, net, devices: list[AbstractDevice], time_horizon, *, parameters=None, nu_star=None
    ):
        if parameters is None:
            parameters = [{} for _ in devices]

        # Initialize
        st = self.initialize_solver(net, devices, time_horizon)
        history = self.initialize_history()

        for iteration in range(self.num_iterations):
            # (1) Device proximal updates
            st = self.device_updates(st, devices, parameters)

            # (2) Update averages and residuals
            last_avg_phase = st.avg_phase
            last_resid_power = st.resid_power
            st = self.update_averages_and_residuals(st, net, devices, time_horizon)

            # (3) Update scaled prices
            st = self.price_updates(st, net, devices, time_horizon)

            # (4) Hisory, convergence checks, numerical checks
            if self.track_objective:
                st = st.update(objective=self.compute_objective(st, devices))

            self.update_history(history, st, last_avg_phase, last_resid_power, nu_star)

            if self.has_converged(st):
                break  # Quit early

            if self.safe_mode:
                self.dimension_checks(st, net, devices, time_horizon)
                self.numerical_checks(st, net, devices, time_horizon)

        return st, history

    # ====
    # Update Rules
    # ====

    def device_updates(self, st: ADMMState, devices, parameters):
        rho_power, rho_angle = self.get_rho()

        for i, dev in enumerate(devices):
            set_p = self.set_power(dev, i, st)
            set_v = self.set_phase(dev, i, st)

            p, v = dev.admm_prox_update(rho_power, rho_angle, set_p, set_v, **parameters[i])
            st.power[i] = p
            st.phase[i] = v

        return st

    def set_power(self, dev: AbstractDevice, dev_index: int, st: ADMMState):
        return [
            p - Ai.T @ (st.avg_power + st.dual_power)
            for p, Ai in zip(st.power[dev_index], dev.incidence_matrix)
        ]

    def set_phase(self, dev: AbstractDevice, dev_index: int, st: ADMMState):
        if st.dual_phase[dev_index] is None:
            return None
        else:
            return [
                Ai.T @ st.avg_phase - v
                for v, Ai in zip(st.dual_phase[dev_index], dev.incidence_matrix)
            ]

    def update_averages_and_residuals(self, st: ADMMState, net, devices, time_horizon):
        # Note: it's important to do this in two steps so that the correct averages
        # are used to calculate residuals.
        st = st.update(
            avg_power=dc_average(st.power, net, devices, time_horizon, st.num_terminals),
            avg_phase=ac_average(st.phase, net, devices, time_horizon, st.num_ac_terminals),
        )
        st = st.update(
            resid_power=get_terminal_residual(st.power, st.avg_power, devices),
            resid_phase=get_terminal_residual(st.phase, st.avg_phase, devices),
        )
        return st

    def price_updates(self, st: ADMMState, net, devices, time_horizon):
        return st.update(
            dual_power=st.dual_power + st.avg_power,
            dual_phase=nested_add(st.dual_phase, st.resid_phase),
        )

    # ====
    # History, numerical checks, etc
    # ====

    def dimension_checks(self, st: ADMMState, net, devices, time_horizon):
        num_devices = len(devices)
        num_nodes = net.num_nodes

        assert len(st.power) == num_devices
        assert len(st.phase) == num_devices
        assert len(st.dual_phase) == num_devices
        assert st.dual_power.shape == (num_nodes, time_horizon)

        return True

    def numerical_checks(self, st: ADMMState, net, devices, time_horizon):
        # Dual phases should average to zero
        avg_dual_phase = ac_average(st.dual_phase, net, devices, time_horizon, st.num_ac_terminals)
        np.testing.assert_allclose(nested_norm(avg_dual_phase), 0.0, atol=1e-8)
        return True

    def compute_objective(self, st: ADMMState, devices: list[AbstractDevice]):
        # TODO Incorporate local variables
        costs = [d.operation_cost(st.power[i], st.phase[i], None) for i, d in enumerate(devices)]
        return sum(costs)

    def has_converged(self, st: ADMMState):
        return False

    def update_history(
        self, history: ADMMState, st: ADMMState, last_avg_phase, last_resid_power, nu_star
    ):
        history.objective += [st.objective]
        p = self.resid_norm

        # Primal/dual residuals
        history = self.update_primal_residuals(history, st)
        history = self.update_dual_residuals(history, st, last_resid_power, last_avg_phase)

        if nu_star is not None:
            history.price_error += [
                np.linalg.norm((st.dual_power * self.rho_power - nu_star).ravel(), p)
            ]

        return history

    def update_primal_residuals(self, history: ADMMState, st: ADMMState):
        p = self.resid_norm
        history.power += [np.linalg.norm(st.avg_power.ravel(), p)]
        history.phase += [nested_norm(st.resid_phase, p)]
        return history

    def update_dual_residuals(
        self, history: ADMMState, st: ADMMState, last_resid_power, last_avg_phase
    ):
        p = self.resid_norm

        dual_resid_power = nested_subtract(st.resid_power, last_resid_power, self.rho_power)
        history.dual_power += [nested_norm(dual_resid_power, p)]
        history.dual_phase += [
            np.linalg.norm((st.avg_phase - last_avg_phase).ravel() * self.rho_angle, p)
        ]

        return history

    # ====
    # Initialization
    # ====

    def initialize_history(self) -> ADMMState:
        history = ADMMState(
            num_terminals=None,
            num_ac_terminals=None,
            power=[],
            phase=[],
            dual_power=[],
            dual_phase=[],
            objective=[],
        )
        history.price_error = []
        return history

    def initialize_solver(self, net, devices, time_horizon) -> ADMMState:
        num_terminals = get_num_terminals(net, devices)
        num_ac_terminals = get_num_terminals(net, devices, only_ac=True)

        power_var = [d.admm_initialize_power_variables(time_horizon) for d in devices]
        phase_var = [d.admm_initialize_angle_variables(time_horizon) for d in devices]

        power_dual = dc_average(power_var, net, devices, time_horizon, num_terminals)
        phase_dual = [d.admm_initialize_angle_variables(time_horizon) for d in devices]

        power_bar = dc_average(power_var, net, devices, time_horizon, num_terminals)
        theta_bar = ac_average(phase_var, net, devices, time_horizon, num_ac_terminals)

        theta_tilde = get_terminal_residual(phase_var, theta_bar, devices)
        power_tilde = get_terminal_residual(power_var, power_bar, devices)

        return ADMMState(
            num_terminals=num_terminals,
            num_ac_terminals=num_ac_terminals,
            power=power_var,
            phase=phase_var,
            dual_power=power_dual,
            dual_phase=phase_dual,
            avg_power=power_bar,
            avg_phase=theta_bar,
            resid_power=power_tilde,
            resid_phase=theta_tilde,
        )
