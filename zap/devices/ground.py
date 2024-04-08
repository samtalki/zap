import torch
import numpy as np
import scipy.sparse as sp

from collections import namedtuple
from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray

from zap.util import replace_none
from .abstract import AbstractDevice, make_dynamic

GroundData = namedtuple(
    "GroundData",
    [
        "voltage",
    ],
)


@dataclass(kw_only=True)
class Ground(AbstractDevice):
    """A single-node device that fixes voltage phase angle."""

    num_nodes: int
    terminal: NDArray
    voltage: Optional[NDArray] = None

    def __post_init__(self):
        self.voltage = make_dynamic(replace_none(self.voltage, np.zeros(self.num_devices)))

    @property
    def terminals(self):
        return self.terminal

    @property
    def is_ac(self):
        return True

    @property
    def time_horizon(self):
        return 0  # Static device

    def _device_data(self):
        return GroundData(self.voltage)

    def equality_constraints(self, power, angle, local_variables, la=np):
        data = self.device_data(la=la)
        return [
            power[0],
            angle[0] - data.voltage,
        ]

    def inequality_constraints(self, power, angle, local_variables, la=np):
        return []

    def operation_cost(self, power, angle, local_variables, la=np):
        if la == torch:
            return la.zeros(1)
        else:
            return 0.0

    def _equality_matrices(self, equalities, la=np):
        size = equalities[0].power[0].shape[1]

        equalities[0].power[0] += sp.eye(size)
        equalities[1].angle[0] += sp.eye(size)
        return equalities

    def _inequality_matrices(self, inequalities, la=np):
        return inequalities

    def scale_costs(self, scale):
        pass

    def scale_power(self, scale):
        pass

    def admm_initialize_power_variables(self, time_horizon: int):
        return [np.zeros((self.num_devices, time_horizon))]

    def admm_initialize_angle_variables(self, time_horizon: int):
        return [np.zeros((self.num_devices, time_horizon))]

    def admm_prox_update(
        self, rho_power, rho_angle, power, angle, power_weights=None, angle_weights=None, la=np
    ):
        data = self.device_data(la=la)

        # Problem is
        #     min_p    {p = 0} + {a = voltage} + (rho/2) || p - power ||_2^2 + (rho/2) || a - angle ||_2^2
        # Solution is just
        #     p = 0, a = voltage

        p = np.zeros_like(power[0])
        a = np.zeros_like(angle[0]) + data.voltage

        return [p], [a]
