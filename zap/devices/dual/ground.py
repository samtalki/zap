import numpy as np

from zap.devices.ground import Ground


class DualGround(Ground):
    def __init__(self, ground: Ground, **kwargs):
        self.primal = ground

        self.num_nodes = ground.num_nodes
        self.terminal = ground.terminal
        self.voltage = ground.voltage

    def equality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return []

    def inequality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return []

    def operation_cost(self, power, angle, _, la=np, envelope=None):
        mu = angle[0]
        v = self.voltage

        return la.sum(la.multiply(mu, v))
