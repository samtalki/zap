import dataclasses
import numpy as np

from zap.devices.abstract import AbstractDevice


def nested_map(f, *args, none_value=None):
    return [
        (none_value if arg_dev[0] is None else [f(*arg_dt) for arg_dt in zip(*arg_dev)])
        for arg_dev in zip(*args)
    ]


def nested_add(x1, x2, alpha=None):
    if alpha is None:
        return nested_map(lambda x, y: x + y, x1, x2)
    else:
        return nested_map(lambda x, y: alpha * (x + y), x1, x2)


def nested_subtract(x1, x2, alpha=None):
    if alpha is None:
        return nested_map(lambda x, y: x - y, x1, x2)
    else:
        return nested_map(lambda x, y: alpha * (x - y), x1, x2)


def nested_norm(data, p=None):
    mini_norms = [
        ([0.0] if x_dev is None else [np.linalg.norm(x.ravel(), p) for x in x_dev])
        for x_dev in data
    ]
    return np.linalg.norm(np.concatenate(mini_norms), p)


def get_discrep(power1, power2):
    discreps = [
        ([0.0] if p_admm is None else [np.linalg.norm(p1 - p2, 1) for p1, p2 in zip(p_admm, p_cvx)])
        for p_admm, p_cvx in zip(power1, power2)
    ]
    return np.sum(np.concatenate(discreps))


def get_num_terminals(net, devices, only_ac=False):
    terminal_counts = np.zeros(net.num_nodes)
    for d in devices:
        if only_ac and (not d.is_ac):
            continue

        values, counts = np.unique(d.terminals, return_counts=True)
        for t, c in zip(values, counts):
            terminal_counts[t] += c

    return np.expand_dims(terminal_counts, 1)


def get_nodal_average(
    powers, net, devices, time_horizon, num_terminals=None, only_ac=False, check_connections=True
):
    if num_terminals is None:
        num_terminals = get_num_terminals(net, devices, only_ac=only_ac)

    if check_connections:
        assert np.all(num_terminals > 0)
    else:
        num_terminals = np.maximum(num_terminals, 1e-8)

    average_x = np.zeros((net.num_nodes, time_horizon))

    for dev, x_dev in zip(devices, powers):
        if x_dev is None:
            continue
        for A_dt, x_dt in zip(dev.incidence_matrix, x_dev):
            average_x += A_dt @ x_dt

    return np.divide(average_x, num_terminals)


def get_terminal_residual(angles, average_angle, devices):
    residuals = [
        None
        if a_dev is None
        else [a_dt - A_dt.T @ average_angle for a_dt, A_dt in zip(a_dev, dev.incidence_matrix)]
        for a_dev, dev in zip(angles, devices)
    ]

    return residuals


def dc_average(x, net, devices, time_horizon, num_terminals):
    return get_nodal_average(x, net, devices, time_horizon, num_terminals)


def ac_average(x, net, devices, time_horizon, num_ac_terminals):
    return get_nodal_average(
        x, net, devices, time_horizon, num_ac_terminals, only_ac=True, check_connections=False
    )


@dataclasses.dataclass
class ADMMState:
    num_terminals: object
    num_ac_terminals: object
    power: object
    phase: object
    dual_power: object
    dual_phase: object
    avg_power: object = None
    avg_phase: object = None
    resid_power: object = None
    resid_phase: object = None
    objective: object = None

    def update(self, **kwargs):
        """Return a new state with fields updated."""
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass
class ADMMSolver:
    """A simple class that stores ADMM solver parameters and exposes a solve function."""

    num_iterations: int
    rho_power: float
    rho_angle: float = None
    atol: float = 0.0
    rtol: float = 1e-3
    resid_norm: object = None
    safe_mode: bool = False
    track_objective: bool = True

    def solve(
        self, net, devices: list[AbstractDevice], time_horizon, *, parameters=None, nu_star=None
    ):
        if parameters is None:
            parameters = [{} for _ in devices]

        # Initialize
        st = self.initialize_solver(net, devices, time_horizon)
        history = self.initialize_history()

        rho_power = self.rho_power
        rho_angle = self.rho_angle

        if rho_angle is None:
            rho_angle = rho_power

        for iteration in range(self.num_iterations):
            # (1) Device proximal updates
            for i, dev in enumerate(devices):
                set_p = self.set_power(dev, st.power[i], st)
                set_v = self.set_phase(dev, st.dual_phase[i], st)

                p, v = dev.admm_prox_update(rho_power, rho_angle, set_p, set_v, **parameters[i])
                st.power[i] = p
                st.phase[i] = v

            # (2) Update averages and residuals
            last_avg_phase = st.avg_phase
            last_resid_power = st.resid_power

            # Note: it's important to do this in two steps so that the correct averages
            # are used to calculate residuals.
            st = st.update(
                avg_power=dc_average(st.power, net, devices, time_horizon, st.num_terminals),
                avg_phase=ac_average(st.phase, net, devices, time_horizon, st.num_ac_terminals),
            )
            st = st.update(
                resid_power=get_terminal_residual(st.power, st.avg_power, devices),
                resid_phase=get_terminal_residual(st.phase, st.avg_phase, devices),
            )

            # (3) Update scaled prices
            st = st.update(
                dual_power=st.dual_power + st.avg_power,
                dual_phase=nested_add(st.dual_phase, st.resid_phase),
            )

            # (4) Update history
            if self.track_objective:
                op_costs = [
                    d.operation_cost(st.power[i], st.phase[i], None) for i, d in enumerate(devices)
                ]
                st = st.update(objective=sum(op_costs))

            self.update_history(history, st, last_avg_phase, last_resid_power, nu_star)

            # (5) Convergence check

            # (Optional) Run numerical checks
            if self.safe_mode:
                # Dual phases should average to zero
                avg_dual_phase = ac_average(
                    st.dual_phase, net, devices, time_horizon, st.num_ac_terminals
                )
                # print(iteration, np.linalg.norm(avg_dual_phase))
                np.testing.assert_allclose(avg_dual_phase, 0.0, atol=1e-8)

        return st, history

    def set_power(self, dev: AbstractDevice, dev_power, st: ADMMState):
        return [
            p - Ai.T @ (st.avg_power + st.dual_power)
            for p, Ai in zip(dev_power, dev.incidence_matrix)
        ]

    def set_phase(self, dev: AbstractDevice, dev_dual_phase, st: ADMMState):
        if dev_dual_phase is None:
            return None
        else:
            return [Ai.T @ st.avg_phase - v for v, Ai in zip(dev_dual_phase, dev.incidence_matrix)]

    def update_history(
        self, history: ADMMState, st: ADMMState, last_avg_phase, last_resid_power, nu_star
    ):
        history.objective += [st.objective]
        p = self.resid_norm

        # Primal residuals
        history.power += [np.linalg.norm(st.avg_power.ravel(), p)]
        history.phase += [nested_norm(st.resid_phase, p)]

        # Dual residuals
        dual_resid_power = nested_subtract(st.resid_power, last_resid_power, self.rho_power)
        history.dual_power += [nested_norm(dual_resid_power, p)]
        history.dual_phase += [
            np.linalg.norm((st.avg_phase - last_avg_phase).ravel() * self.rho_angle, p)
        ]

        if nu_star is not None:
            history.price_error += [
                np.linalg.norm((st.dual_power * self.rho_power - nu_star).ravel(), p)
            ]

        return history

    def initialize_history(self) -> ADMMState:
        history = ADMMState(
            num_terminals=None,
            num_ac_terminals=None,
            power=[],
            phase=[],
            dual_power=[],
            dual_phase=[],
            objective=[],
        )
        history.price_error = []
        return history

    def initialize_solver(self, net, devices, time_horizon) -> ADMMState:
        num_terminals = get_num_terminals(net, devices)
        num_ac_terminals = get_num_terminals(net, devices, only_ac=True)

        power_var = [d.admm_initialize_power_variables(time_horizon) for d in devices]
        phase_var = [d.admm_initialize_angle_variables(time_horizon) for d in devices]

        power_dual = dc_average(power_var, net, devices, time_horizon, num_terminals)
        phase_dual = [d.admm_initialize_angle_variables(time_horizon) for d in devices]

        power_bar = dc_average(power_var, net, devices, time_horizon, num_terminals)
        theta_bar = ac_average(phase_var, net, devices, time_horizon, num_ac_terminals)

        theta_tilde = get_terminal_residual(phase_var, theta_bar, devices)
        power_tilde = get_terminal_residual(power_var, power_bar, devices)

        return ADMMState(
            num_terminals=num_terminals,
            num_ac_terminals=num_ac_terminals,
            power=power_var,
            phase=phase_var,
            dual_power=power_dual,
            dual_phase=phase_dual,
            avg_power=power_bar,
            avg_phase=theta_bar,
            resid_power=power_tilde,
            resid_phase=theta_tilde,
        )
