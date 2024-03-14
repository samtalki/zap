import torch
import numpy as np

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
        self.voltage = make_dynamic(
            replace_none(self.voltage, np.zeros(self.num_devices))
        )

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

    def equality_constraints(self, power, angle, local_variable, la=np):
        data = self.device_data(la=la)
        return [
            power[0],
            angle[0] - data.voltage,
        ]

    def inequality_constraints(self, power, angle, local_variable, la=np):
        return []

    def operation_cost(self, power, angle, local_variable, la=np):
        if la == torch:
            return la.zeros(1)
        else:
            return 0.0
