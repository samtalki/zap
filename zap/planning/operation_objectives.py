import numpy as np
import torch

from zap.network import DispatchOutcome, PowerNetwork
from zap.devices.abstract import AbstractDevice


class AbstractOperationObjective:
    """Abstract implementation of operation objectives."""

    def __call__(self, y: DispatchOutcome, parameters=None, use_torch=False):
        return self.forward(y, parameters=parameters, use_torch=use_torch)

    def forward(self, y: DispatchOutcome, parameters=None, use_torch=False):
        raise NotImplementedError

    @property
    def is_convex(self):
        return False

    @property
    def is_linear(self):
        return False


class DispatchCostObjective(AbstractOperationObjective):
    """Cost of the dispatch outcome."""

    def __init__(self, net: PowerNetwork, devices):
        self.net = net
        self.devices = devices

    def forward(self, y: DispatchOutcome, parameters=None, use_torch=False):
        if use_torch:
            la = torch
        else:
            la = np

        return self.net.operation_cost(
            self.devices, y.power, y.angle, y.local_variables, parameters=parameters, la=la
        )

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False


class EmissionsObjective(AbstractOperationObjective):
    """Total emissions of the dispatch outcome."""

    def __init__(self, devices: list[AbstractDevice]):
        self.devices = devices

    def forward(self, y: DispatchOutcome, parameters=None, use_torch=False):
        if use_torch:
            la = torch
        else:
            la = np

        emissions = [
            d.get_emissions(p, **param, la=la)
            for p, d, param in zip(y.power, self.devices, parameters)
        ]

        return sum(emissions)

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return True
