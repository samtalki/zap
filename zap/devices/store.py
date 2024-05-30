import torch
import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from typing import Optional
from collections import namedtuple
from numpy.typing import NDArray

from zap.devices.abstract import AbstractDevice, make_dynamic

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

    def scale_costs(self, scale):
        self.linear_cost /= scale
        if self.quadratic_cost is not None:
            self.quadratic_cost /= scale
        if self.capital_cost is not None:
            self.capital_cost /= scale

    def scale_power(self, scale):
        self.power_capacity /= scale
        if self.min_power_capacity is not None:
            self.min_power_capacity /= scale
        if self.max_power_capacity is not None:
            self.max_power_capacity /= scale

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return BatteryVariable(
            cp.Variable((self.num_devices, time_horizon + 1)),
            cp.Variable((self.num_devices, time_horizon)),
            cp.Variable((self.num_devices, time_horizon)),
        )

    def equality_constraints(self, power, angle, state, power_capacity=None, la=np, envelope=None):
        power_capacity = self.parameterize(power_capacity=power_capacity, la=la)

        if not isinstance(state, BatteryVariable):
            state = BatteryVariable(*state)

        T = power[0].shape[1]
        energy_capacity = la.multiply(power_capacity, self.duration)

        soc_evolution = (
            state.energy[:, :-1]
            + la.multiply(state.charge, self.charge_efficiency)
            - state.discharge
        )
        return [
            power[0] - (state.discharge - state.charge),
            state.energy[:, 1:] - soc_evolution,
            state.energy[:, 0:1] - la.multiply(self.initial_soc, energy_capacity),
            state.energy[:, T : (T + 1)] - la.multiply(self.final_soc, energy_capacity),
        ]

    def inequality_constraints(
        self, power, angle, state, power_capacity=None, la=np, envelope=None
    ):
        power_capacity = self.parameterize(power_capacity=power_capacity, la=la)

        if not isinstance(state, BatteryVariable):
            state = BatteryVariable(*state)

        energy_capacity = la.multiply(power_capacity, self.duration)

        return [
            -state.energy,
            state.energy - energy_capacity,
            -state.charge,
            state.charge - power_capacity,
            -state.discharge,
            state.discharge - power_capacity,
        ]

    def operation_cost(self, power, angle, state, power_capacity=None, la=np, envelope=None):
        if state is None:
            return 0.0

        if not isinstance(state, BatteryVariable):
            state = BatteryVariable(*state)

        cost = la.sum(la.multiply(self.linear_cost, state.discharge))
        if self.quadratic_cost is not None:
            cost += la.sum(la.multiply(self.quadratic_cost, la.square(state.discharge)))

        return cost

    # ====
    # DIFFERENTIATION
    # ====

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
        # Dimensions
        size = equalities[0].power[0].shape[1]
        time_horizon = int(size / self.num_devices)
        shaped_zeros = np.zeros((self.num_devices, time_horizon))

        # Power balance
        equalities[0].power[0] += sp.eye(size)
        equalities[0].local_variables[1] += sp.eye(size)
        equalities[0].local_variables[2] += -sp.eye(size)

        # SOC evolution
        alpha = shaped_zeros + self.charge_efficiency
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

    # ====
    # PLANNING FUNCTIONS
    # ====

    def get_investment_cost(self, power_capacity=None, la=np):
        power_capacity = self.parameterize(power_capacity=power_capacity, la=la)

        if self.capital_cost is None or power_capacity is None:
            return 0.0

        # Get original nominal capacity and capital cost
        # Nominal capacity isn't passed here because we want to use the original value
        pnom_min = self.power_capacity
        capital_cost = self.capital_cost

        return la.sum(la.multiply(capital_cost, (power_capacity - pnom_min)))

    # ====
    # ADMM FUNCTIONS
    # ====
    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        power_capacity=None,
        power_weights=None,
        angle_weights=None,
        window=None,
        inner_weight=1.0,
        inner_over_relaxation=1.0,
        inner_iterations=25,
    ):
        power_capacity = self.parameterize(power_capacity=power_capacity)
        N, full_time_horizon = power[0].shape
        T = full_time_horizon if window is None else window  # Window size
        num_scenarios = full_time_horizon // T

        machine, dtype = power[0].device, power[0].dtype

        assert angle is None
        assert full_time_horizon % T == 0

        # Fixed data - constant between solves
        if not hasattr(self, "admm_data"):
            self.admm_data = battery_prox_data(self, T, rho_power, power[0], inner_weight)

        # Variable data that changes between solves
        if getattr(self, "has_changed", False):
            self.has_changed = False

            smax = torch.multiply(power_capacity, self.duration)
            gamma1 = torch.multiply(self.initial_soc, smax)
            gammaT = torch.multiply(self.final_soc, smax)
            ymin, ymax = get_ymin_ymax(T, power_capacity, smax, gamma1, gammaT, machine, dtype)

            self.temp_data = (smax, gamma1, gammaT, ymin, ymax)

        K_inv, zero_nu = self.admm_data
        smax, gamma1, gammaT, ymin, ymax = self.temp_data

        # Changes every proximal evaluation
        zT = power[0].reshape(-1, num_scenarios, T, 1)
        rhs = K_rhs_fixed(self, T, rho_power, zT, machine)

        # Initialize
        x = torch.zeros((N, num_scenarios, 3 * T + 1, 1), device=machine, dtype=dtype)
        y = torch.zeros(x.shape, device=machine, dtype=dtype)
        u = torch.zeros(x.shape, device=machine, dtype=dtype)

        y[:, :, :T, :] = torch.relu(-zT)
        y[:, :, T : (2 * T), :] = torch.relu(zT)
        y[:, :, (2 * T) :, :] = gamma1.reshape(-1, 1, 1, 1)

        # Solve ADMM
        for iter in range(inner_iterations):
            x, y, u = battery_prox_inner(
                x, y, u, rhs, zero_nu, K_inv, ymin, ymax, inner_weight, inner_over_relaxation
            )

        # Extract results
        c = x[:, :, :T, 0].reshape(N, -1)
        d = x[:, :, T : (2 * T), 0].reshape(N, -1)
        # s = x[:, :, (2 * T) :, 0].reshape(N, -1)

        return [d - c], None


# ====
# ADMM UTILITY FUNCTIONS
# ====


def difference_matrix(T, machine=None):
    # Should return a (T, T+1) matrix where
    # D = [-1 1 0 0]
    #     [0 -1 1 0]
    #     [0 0 -1 1]
    # for T = 3

    D1 = torch.eye(T + 1, device=machine)[0:T, :]
    D2 = torch.eye(T + 1, device=machine)[1:, :]

    return D2 - D1


def b_vector(device: Battery, T, machine=None):
    # TODO - Support multiple costs
    alpha = device.linear_cost[0]
    return torch.vstack(
        [
            alpha * torch.ones((T, 1), device=machine),
            torch.zeros((2 * T + 1, 1), device=machine),
        ]
    )


def C_matrix(device: Battery, T, machine=None):
    # TODO - Support multiple charge efficiencies
    beta = device.charge_efficiency[0]
    D = difference_matrix(T, machine)
    Id = torch.eye(T, device=machine)

    return torch.hstack([-beta * Id, Id, D])


def A_matrix(T, machine=None):
    Id = torch.eye(T, device=machine)
    return torch.hstack([-Id, Id, torch.zeros(T, T + 1, device=machine)])


def K_matrix(device: Battery, T, rho, w, machine=None):
    A = A_matrix(T, machine)
    C = C_matrix(device, T, machine)
    Id = torch.eye(3 * T + 1, device=machine)

    dKdx = rho * (A.T @ A) + w * Id

    row1 = torch.hstack([dKdx, C.T])
    row2 = torch.hstack([C, torch.zeros(T, T, device=machine)])

    return torch.vstack([row1, row2])


def K_rhs_fixed(device: Battery, T, rho, z, machine=None):
    # rho * A.T @ z - b
    A = A_matrix(T, machine)
    b = b_vector(device, T, machine)
    return rho * torch.matmul(A.T, z) - b


@torch.jit.script
def battery_prox_inner(x, y, u, rhs, zero_nu, K_inv, ymin, ymax, w: float, alpha: float = 1.0):
    # x update
    rhs_var = rhs + w * (y - u)
    full_rhs = torch.cat([rhs_var, zero_nu], dim=2)
    x = (K_inv @ full_rhs)[:, :, : x.shape[2], :]

    # over relaxation step
    xp = alpha * x + (1 - alpha) * y

    # y update
    y = torch.clip(xp + u, min=ymin, max=ymax)

    # u update
    u += xp - y

    return x, y, u


def battery_prox_data(device: Battery, T: int, rho, z, weight=1.0):
    machine = z.device

    T_full = z.shape[1]
    num_scenarios = T_full // T
    assert T * num_scenarios == T_full

    # Fixed data that does not change between solves
    K = K_matrix(device, T, rho, weight, machine)
    K_inv = torch.linalg.inv(K)

    # zT and rhs are just created to get the dimensions for zero_nu
    zT = z.reshape(-1, num_scenarios, T, 1)
    rhs = K_rhs_fixed(device, T, rho, zT, machine)
    zero_nu = torch.zeros((rhs.shape[0], rhs.shape[1], T, rhs.shape[3]), device=machine)

    return K_inv, zero_nu


def get_ymin_ymax(T, pmax, smax, gamma1, gammaT, machine=None, dtype=None):
    ymax = torch.hstack([pmax.expand(-1, 2 * T), smax.expand(-1, T + 1)])
    ymin = torch.zeros(ymax.shape, device=machine, dtype=dtype)

    ymin[:, 2 * T] = gamma1[:, 0]
    ymax[:, 2 * T] = gamma1[:, 0]
    ymin[:, -1] = gammaT[:, 0]
    ymax[:, -1] = gammaT[:, 0]

    ymin = ymin.reshape(ymin.shape[0], 1, ymin.shape[1], 1)
    ymax = ymax.reshape(ymax.shape[0], 1, ymax.shape[1], 1)

    return ymin, ymax
