import cvxpy as cp
import numpy as np
from copy import deepcopy

import zap.dual
from zap.network import DispatchOutcome
from zap.planning.problem import PlanningProblem


class RelaxedPlanningProblem:
    def __init__(
        self,
        problem: PlanningProblem,
        inf_value=100.0,
        max_price=100.0,
        solver=cp.MOSEK,
        sd_tolerance=1.0,
        solver_kwargs={"verbose": False, "accept_unknown": True},
    ):
        self.problem = deepcopy(problem)
        self.inf_value = inf_value
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.sd_tolerance = sd_tolerance
        self.max_price = max_price

    def setup_parameters(self, **kwargs):
        return self.problem.layer.setup_parameters(**kwargs)

    def model_outer_problem(self):
        """Define outer variables, constraints, and costs."""
        network_parameters = {
            p: cp.Variable(lower.shape) for p, lower in self.problem.lower_bounds.items()
        }

        lower_bounds = {}
        upper_bounds = {}

        for p in network_parameters.keys():
            lower = self.problem.lower_bounds[p]
            upper = self.problem.upper_bounds[p]

            # Replace infs with inf_value times max value
            inf_param = self.inf_value * np.max(lower)
            upper = np.where(upper == np.inf, inf_param, upper)

            lower_bounds[p] = lower
            upper_bounds[p] = upper

        investment_objective = self.problem.investment_objective(la=cp, **network_parameters)

        return network_parameters, lower_bounds, upper_bounds, investment_objective

    def solve(self):
        """Solve strong-duality relaxed planning problem."""

        # Define outer variables, constraints, and costs
        net_params, lower, upper, investment_objective = self.model_outer_problem()
        box_constraints = [lower[p] <= net_params[p] for p in sorted(net_params.keys())]
        box_constraints += [net_params[p] <= upper[p] for p in sorted(net_params.keys())]

        # Define primal and dual problems
        net, devices = self.problem.layer.network, self.problem.layer.devices
        dual_devices = zap.dual.dualize(devices, max_price=self.max_price)

        parameters = self.setup_parameters(**net_params)
        envelope_constraints = []
        envelope_variables = []

        primal_costs, primal_constraints, primal_data = net.model_dispatch_problem(
            devices,
            self.problem.time_horizon,
            dual=False,
            parameters=parameters,
            envelope=(envelope_variables, envelope_constraints),
            lower_param=self.setup_parameters(**lower),
            upper_param=self.setup_parameters(**upper),
        )
        dual_costs, dual_constraints, dual_data = net.model_dispatch_problem(
            dual_devices,
            self.problem.time_horizon,
            dual=True,
            parameters=parameters,
            envelope=(envelope_variables, envelope_constraints),
            lower_param=self.setup_parameters(**lower),
            upper_param=self.setup_parameters(**upper),
        )

        # Define strong duality coupling constraint
        sd_constraint = cp.sum(primal_costs) + cp.sum(dual_costs) <= self.sd_tolerance

        # Define operation objective in terms of primal and dual variables
        y = DispatchOutcome(
            power=primal_data["power"],
            angle=primal_data["angle"],
            global_angle=primal_data["global_angle"],
            local_variables=primal_data["local_variables"],
            prices=dual_data["global_angle"],
            phase_duals=dual_data["power"],
            local_equality_duals=None,
            local_inequality_duals=None,
        )

        # TODO Incorporate true parameters
        operation_objective = self.problem.operation_objective(y, parameters=parameters, la=cp)

        # Create full problem and solve
        problem = cp.Problem(
            cp.Minimize(investment_objective + operation_objective),
            box_constraints + [sd_constraint] + list(primal_constraints) + list(dual_constraints),
        )
        problem.solve(solver=self.solver, **self.solver_kwargs)

        return {
            "network_parameters": net_params,
            "lower_bounds": lower,
            "upper_bounds": upper,
            "box_constraints": box_constraints,
            "investment_objective": investment_objective,
            "problem": problem,
            "sd_constraint": sd_constraint,
            "primal_costs": primal_costs,
            "dual_costs": dual_costs,
            "primal_constraints": primal_constraints,
            "dual_constraints": dual_constraints,
            "primal_data": primal_data,
            "dual_data": dual_data,
            "operation_objective": operation_objective,
        }
