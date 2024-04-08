import dataclasses
import numpy as np

from zap.devices.abstract import AbstractDevice
from zap.admm.basic_solver import ADMMSolver, ADMMState
from zap.admm.util import (
    nested_add,
    nested_subtract,
    dc_average,
    ac_average,
    get_terminal_residual,
    nested_norm,
)


@dataclasses.dataclass
class ExtendedADMMState(ADMMState):
    copy_power: object = None
    copy_phase: object = None


@dataclasses.dataclass
class WeightedADMMSolver(ADMMSolver):
    """Stores weighted ADMM solver parameters and exposes a solve function."""

    # TODO - Device updates must include diagonal scaling matrices
    # TODO - z / ksi updates must include diagonal scaling matrices

    def initialize_solver(self, net, devices, time_horizon) -> ExtendedADMMState:
        st = super().initialize_solver(net, devices, time_horizon)

        st = ExtendedADMMState(
            copy_power=st.power.copy(), copy_phase=st.phase.copy(), **st.__dict__
        )
        return st.update(
            dual_power=st.power.copy(),  # This needs to have terminal dimensions, not node dimensions
        )

    def set_power(self, dev: AbstractDevice, dev_index: int, st: ExtendedADMMState):
        return [z - omega for z, omega in zip(st.copy_power[dev_index], st.dual_power[dev_index])]

    def set_phase(self, dev: AbstractDevice, dev_index: int, st: ExtendedADMMState):
        if st.dual_phase[dev_index] is None:
            return None
        else:
            return [ksi - nu for ksi, nu in zip(st.copy_phase[dev_index], st.dual_phase[dev_index])]

    def price_updates(self, st: ExtendedADMMState):
        return st.update(
            dual_power=nested_add(st.dual_power, nested_subtract(st.power, st.copy_power)),
            dual_phase=nested_add(st.dual_phase, nested_subtract(st.phase, st.copy_phase)),
        )

    def update_averages_and_residuals(self, st: ExtendedADMMState, net, devices, time_horizon):
        st = super().update_averages_and_residuals(st, net, devices, time_horizon)

        avg_dual_power = dc_average(st.dual_power, net, devices, time_horizon, st.num_terminals)
        resid_dual_power = get_terminal_residual(st.dual_power, avg_dual_power, devices)

        avg_dual_phase = ac_average(st.dual_phase, net, devices, time_horizon, st.num_ac_terminals)

        st = st.update(
            copy_power=st.resid_power + resid_dual_power,
            copy_phase=st.avg_phase + avg_dual_phase,
        )
        return st

    def update_history(
        self, history: ADMMState, st: ADMMState, last_avg_phase, last_resid_power, nu_star
    ):
        history.objective += [st.objective]

        # Primal/dual residuals
        history = self.update_primal_residuals(history, st)
        history = self.update_dual_residuals(history, st, last_resid_power, last_avg_phase)

        if nu_star is not None:
            pass  # TODO - Shape problems

        return history
