import numpy as np
import cvxpy as cp

from zap.devices.store import Battery


class DualBattery(Battery):
    def __init__(self, battery: Battery):
        self.primal = battery

        self.num_nodes = battery.num_nodes
        self.terminal = battery.terminal
        self.power_capacity = battery.power_capacity
        self.duration = battery.duration
        self.charge_efficiency = battery.charge_efficiency
        self.initial_soc = battery.initial_soc
        self.final_soc = battery.final_soc
        self.linear_cost = battery.linear_cost
        self.quadratic_cost = battery.quadratic_cost
        self.capital_cost = battery.capital_cost
        self.min_power_capacity = battery.min_power_capacity
        self.max_power_capacity = battery.max_power_capacity

    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return [cp.Variable((self.num_devices, time_horizon))]

    def equality_constraints(self, power, angle, _, power_capacity=None, la=np, envelope=None):
        return []

    def inequality_constraints(self, power, angle, _, power_capacity=None, la=np, envelope=None):
        return []

    def operation_cost(self, power, angle, state, power_capacity=None, la=np, envelope=None):
        data = self.device_data(power_capacity=power_capacity, la=la)
        assert data.quadratic_cost is None

        # Dual variables
        z = power[0]
        lamb = state[0]

        # Parameters
        pmax = data.power_capacity
        smax = pmax * data.duration
        rho = data.charge_efficiency

        # Charge and discharge terms
        c_term = la.maximum(0.0, la.multiply(z - la.multiply(rho, lamb), pmax))
        d_term = la.maximum(0.0, la.multiply(lamb - z - data.linear_cost, pmax))

        # Energy terms
        s1_term = la.multiply(-lamb[:, [0]], data.initial_soc * smax)
        sT_term = la.multiply(lamb[:, [-1]], data.final_soc * smax)

        # Note - this should have (T-1) columns
        s_term = la.maximum(0.0, la.multiply(lamb[:, :-1] - lamb[:, 1:], smax))

        return la.sum(c_term) + la.sum(d_term) + la.sum(s1_term) + la.sum(sT_term) + la.sum(s_term)
