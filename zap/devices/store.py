import numpy as np
import cvxpy as cp

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
