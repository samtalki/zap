import numpy as np
import cvxpy as cp

from typing import Optional
from collections import namedtuple
from numpy.typing import NDArray

from zap.devices.abstract import AbstractDevice, make_dynamic
from zap.util import replace_none

BatteryVariable = namedtuple(
    "BatteryVariable",
    [
        "energy",
        "charge",
        "discharge",
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

    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return BatteryVariable(
            cp.Variable((self.num_devices, time_horizon + 1)),
            cp.Variable((self.num_devices, time_horizon)),
            cp.Variable((self.num_devices, time_horizon)),
        )

    def model_local_constraints(self, power, angle, state, power_capacity=None):
        power_capacity = make_dynamic(replace_none(power_capacity, self.power_capacity))
        energy_capacity = np.multiply(power_capacity, self.duration)

        soc_evolution = (
            state.energy[:, :-1]
            + cp.multiply(state.charge, self.charge_efficiency)
            - state.discharge
        )

        return [
            power[0] == state.charge - state.discharge,
            state.energy[:, 1:] == soc_evolution,
            state.energy[:, 0] == np.multiply(self.initial_soc, energy_capacity),
            state.energy[:, -1] == np.multiply(self.final_soc, energy_capacity),
            state.energy >= 0,
            state.energy <= energy_capacity,
            state.charge >= 0,
            state.charge <= power_capacity,
            state.discharge >= 0,
            state.discharge <= power_capacity,
        ]

    def model_cost(self, power, angle, state, power_capacity=None):
        cost = cp.sum(cp.multiply(self.linear_cost, state.discharge))
        if self.quadratic_cost is not None:
            cost += cp.sum(cp.multiply(self.quadratic_cost, cp.square(state.discharge)))

        return cost
