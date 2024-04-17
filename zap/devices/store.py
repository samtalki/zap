import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from typing import Optional
from collections import namedtuple
from numpy.typing import NDArray

from zap.devices.abstract import AbstractDevice, make_dynamic
from zap.util import replace_none, choose_base_modeler

BatteryVariable = namedtuple(
    "BatteryVariable",
    [
        "energy",
        "charge",
        "discharge",
    ],
)

BatteryData = namedtuple(
    "BatteryData",
    [
        "power_capacity",
        "duration",
        "charge_efficiency",
        "initial_soc",
        "final_soc",
        "linear_cost",
        "quadratic_cost",
        "capital_cost",
    ],
)


class Battery(AbstractDevice):
    """An Injector that stores power between time steps.

    May have a discharge cost.
    """

    def __init__(
        self,
        *,
        num_nodes,
        terminal,
        power_capacity: NDArray,
        duration: NDArray,
        charge_efficiency: Optional[NDArray] = None,
        initial_soc: Optional[NDArray] = None,
        final_soc: Optional[NDArray] = None,
        linear_cost: Optional[NDArray] = None,
        quadratic_cost: Optional[NDArray] = None,
        capital_cost: Optional[NDArray] = None,
        min_power_capacity=None,
        max_power_capacity=None,
    ):
        if linear_cost is None:
            linear_cost = np.zeros(power_capacity.shape)

        if charge_efficiency is None:
            charge_efficiency = np.ones(power_capacity.shape)

        if initial_soc is None:
            initial_soc = 0.5 * np.ones(power_capacity.shape)

        if final_soc is None:
            final_soc = 0.5 * np.ones(power_capacity.shape)

        self.num_nodes = num_nodes
        self.terminal = terminal
        self.power_capacity = make_dynamic(power_capacity)
        self.duration = make_dynamic(duration)
        self.charge_efficiency = make_dynamic(charge_efficiency)
        self.initial_soc = make_dynamic(initial_soc)
        self.final_soc = make_dynamic(final_soc)
        self.linear_cost = make_dynamic(linear_cost)
        self.quadratic_cost = make_dynamic(quadratic_cost)
        self.capital_cost = make_dynamic(capital_cost)
        self.min_power_capacity = make_dynamic(min_power_capacity)
        self.max_power_capacity = make_dynamic(max_power_capacity)

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        return 0  # Static device

    def _device_data(self, power_capacity=None):
        return BatteryData(
            make_dynamic(replace_none(power_capacity, self.power_capacity)),
            self.duration,
            self.charge_efficiency,
            self.initial_soc,
            self.final_soc,
            self.linear_cost,
            self.quadratic_cost,
            self.capital_cost,
        )

    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return BatteryVariable(
            cp.Variable((self.num_devices, time_horizon + 1)),
            cp.Variable((self.num_devices, time_horizon)),
            cp.Variable((self.num_devices, time_horizon)),
        )

    def equality_constraints(self, power, angle, state, power_capacity=None, la=np):
        data = self.device_data(power_capacity=power_capacity, la=la)
        base = choose_base_modeler(la)

        if not isinstance(state, BatteryVariable):
            state = BatteryVariable(*state)

        T = power[0].shape[1]
        energy_capacity = base.multiply(data.power_capacity, data.duration)

        soc_evolution = (
            state.energy[:, :-1]
            + la.multiply(state.charge, data.charge_efficiency)
            - state.discharge
        )
        return [
            power[0] - (state.charge - state.discharge),
            state.energy[:, 1:] - soc_evolution,
            state.energy[:, 0:1] - base.multiply(data.initial_soc, energy_capacity),
            state.energy[:, T : (T + 1)] - base.multiply(data.final_soc, energy_capacity),
        ]

    def inequality_constraints(self, power, angle, state, power_capacity=None, la=np):
        data = self.device_data(power_capacity=power_capacity, la=la)
        base = choose_base_modeler(la)

        if not isinstance(state, BatteryVariable):
            state = BatteryVariable(*state)

        energy_capacity = base.multiply(data.power_capacity, data.duration)

        return [
            -state.energy,
            state.energy - energy_capacity,
            -state.charge,
            state.charge - data.power_capacity,
            -state.discharge,
            state.discharge - data.power_capacity,
        ]

    def operation_cost(self, power, angle, state, power_capacity=None, la=np):
        data = self.device_data(power_capacity=power_capacity, la=la)

        if not isinstance(state, BatteryVariable):
            state = BatteryVariable(*state)

        cost = la.sum(la.multiply(data.linear_cost, state.discharge))
        if data.quadratic_cost is not None:
            cost += la.sum(la.multiply(data.quadratic_cost, la.square(state.discharge)))

        return cost

    def _soc_boundary_matrix(self, num_devices, time_horizon, index=0):
        soc_first = np.zeros((num_devices, time_horizon + 1))
        soc_first[:, index] = 1.0

        cols = sp.diags(soc_first.ravel(), format="coo").col
        rows = np.arange(num_devices)
        values = np.ones(len(rows))
        shape = (num_devices, num_devices * (time_horizon + 1))

        return sp.coo_matrix((values, (rows, cols)), shape=shape)

    def _soc_difference_matrix(self, num_devices, time_horizon):
        empty = np.zeros((num_devices, time_horizon + 1))

        last_soc = empty.copy()
        last_soc[:, :-1] = -1.0

        next_soc = empty.copy()
        next_soc[:, 1:] = 1.0

        c1 = sp.diags(last_soc.ravel(), format="coo")
        c2 = sp.diags(next_soc.ravel(), format="coo")
        r = np.arange(num_devices * time_horizon)

        cols = np.concatenate([c1.col, c2.col])
        rows = np.concatenate([r, r])
        values = np.concatenate([c1.data, c2.data])
        shape = (num_devices * time_horizon, num_devices * (time_horizon + 1))

        return sp.coo_matrix((values, (rows, cols)), shape=shape)

    def _equality_matrices(self, equalities, power_capacity=None, la=np):
        data = self.device_data(power_capacity=power_capacity, la=la)

        # Dimensions
        size = equalities[0].power[0].shape[1]
        time_horizon = int(size / self.num_devices)
        shaped_zeros = np.zeros((self.num_devices, time_horizon))

        # Power balance
        equalities[0].power[0] += sp.eye(size)
        equalities[0].local_variables[1] += -sp.eye(size)
        equalities[0].local_variables[2] += sp.eye(size)

        # SOC evolution
        alpha = shaped_zeros + data.charge_efficiency
        soc_diff = self._soc_difference_matrix(self.num_devices, time_horizon)

        equalities[1].local_variables[0] += soc_diff  # Energy
        equalities[1].local_variables[1] += -sp.diags(alpha.ravel())  # Charging
        equalities[1].local_variables[2] += sp.eye(size)  # Discharging

        # Initial / Final SOC
        equalities[2].local_variables[0] += self._soc_boundary_matrix(
            self.num_devices, time_horizon, index=0
        )
        equalities[3].local_variables[0] += self._soc_boundary_matrix(
            self.num_devices, time_horizon, index=-1
        )

        return equalities

    def _inequality_matrices(self, inequalities, power_capacity=None, la=np):
        size = inequalities[0].power[0].shape[1]
        e_size = inequalities[0].local_variables[0].shape[0]

        inequalities[0].local_variables[0] += -sp.eye(e_size)
        inequalities[1].local_variables[0] += sp.eye(e_size)
        inequalities[2].local_variables[1] += -sp.eye(size)
        inequalities[3].local_variables[1] += sp.eye(size)
        inequalities[4].local_variables[2] += -sp.eye(size)
        inequalities[5].local_variables[2] += sp.eye(size)

        return inequalities

    def scale_costs(self, scale):
        self.linear_cost /= scale
        if self.quadratic_cost is not None:
            self.quadratic_cost /= scale
        if self.capital_cost is not None:
            self.capital_cost /= scale

    def scale_power(self, scale):
        self.power_capacity /= scale

    def get_investment_cost(self, power_capacity=None, la=np):
        # Get original nominal capacity and capital cost
        # Nominal capacity isn't passed here because we want to use the original value
        data = self.device_data(la=la)

        if self.capital_cost is None or power_capacity is None:
            return 0.0

        pnom_min = data.power_capacity
        capital_cost = data.capital_cost

        return la.sum(capital_cost * (power_capacity - pnom_min))
