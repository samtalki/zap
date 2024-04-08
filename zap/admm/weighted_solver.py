import dataclasses
import numpy as np

from zap.devices.abstract import AbstractDevice
from zap.admm.basic_solver import ADMMSolver, ADMMState
from zap.admm.util import (
    nested_add,
    dc_average,
    ac_average,
    get_terminal_residual,
)


@dataclasses.dataclass
class ExtendedADMMState(ADMMState):
    copy_power: object
    copy_phase: object


@dataclasses.dataclass
class WeightedADMMSolver(ADMMSolver):
    """Stores weighted ADMM solver parameters and exposes a solve function."""

    pass

    def initialize_solver(self, net, devices, time_horizon) -> ExtendedADMMState:
        st = super().initialize_solver(net, devices, time_horizon)

        return ExtendedADMMState(
            **st.__dict__, copy_power=st.power.copy(), copy_phase=st.phase.copy()
        )
