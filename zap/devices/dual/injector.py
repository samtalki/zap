import numpy as np

from zap.devices.injector import Injector


class DualInjector(Injector):
    def __init__(self, injector: Injector):
        self.primal = injector

        self.num_nodes = injector.num_nodes
        self.terminal = injector.terminal
        self.min_power = injector.min_power
        self.max_power = injector.max_power
        self.linear_cost = injector.linear_cost
        self.quadratic_cost = injector.quadratic_cost
        self.nominal_capacity = injector.nominal_capacity
        self.capital_cost = injector.capital_cost
        self.emission_rates = injector.emission_rates

    # This isn't helpful because of the way dataclasses define attributes
    # def __getattr__(self, attr):
    #     return getattr(self.primal, attr)

    def equality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return []

    def inequality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return []

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)

        assert data.quadratic_cost is None

        z = power[0]
        pnom = data.nominal_capacity
        pmin = data.min_power
        pmax = data.max_power
        c = data.linear_cost

        f1 = la.multiply(la.multiply(z, pnom), pmin)
        f2 = la.multiply(la.multiply(z, pnom), pmax)
        f2 -= la.multiply(c, la.multiply(pnom, (pmax - pmin)))

        return la.sum(la.maximum(f1, f2))
