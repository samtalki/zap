import torch
import numpy as np

import zap.util as util
from zap.network import PowerNetwork
from zap.devices.abstract import AbstractDevice


class AbstractInvestmentObjective:
    """Abstract implementation of investment objectives."""

    def __call__(self, parameters=None, use_torch=False):
        return self.forward(parameters=parameters, use_torch=use_torch)

    def forward(self, parameters=None, use_torch=False):
        raise NotImplementedError

    @property
    def is_convex(self):
        return False

    @property
    def is_linear(self):
        return False


class InvestmentObjective(AbstractInvestmentObjective):
    """Simple linear investment objective."""

    def __init__(self, net: PowerNetwork, devices: list[AbstractDevice]):
        self.net = net
        self.devices = devices

    def forward(self, parameters=None, use_torch=False):
        parameters = util.expand_params(parameters, self.devices)

        if use_torch:
            la = torch
        else:
            la = np

        costs = [
            d.get_investment_cost(la=la, **param) for d, param in zip(self.devices, parameters)
        ]

        return sum(costs)
