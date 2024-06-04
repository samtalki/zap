import cvxpy as cp
import numpy as np
from copy import deepcopy

import zap.dual
from zap.network import DispatchOutcome
from zap.planning.problem_abstract import AbstractPlanningProblem, StochasticPlanningProblem


class RelaxedPlanningProblem:
    def __init__(
        self,
        problem: AbstractPlanningProblem,
        inf_value=100.0,
        max_price=100.0,
        solver=None,
        sd_tolerance=1.0,
        solver_kwargs={},
    ):
        self.problem = deepcopy(problem)
        self.inf_value = inf_value
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.sd_tolerance = sd_tolerance
        self.max_price = max_price

        if self.solver is None:
            self.solver = self.problem.layer.solver
            self.solver_kwargs = self.problem.layer.solver_kwargs

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

        if isinstance(self.problem, StochasticPlanningProblem):
            inv_func = self.problem.subproblems[0].investment_objective
        else:
            inv_func = self.problem.investment_objective

        investment_objective = inv_func(la=cp, **network_parameters)

        return network_parameters, lower_bounds, upper_bounds, investment_objective

    def solve(self):
        """Solve strong-duality relaxed planning problem."""

        # Define outer variables, constraints, and costs
        net_params, lower, upper, investment_objective = self.model_outer_problem()
        box_constraints = [lower[p] <= net_params[p] for p in sorted(net_params.keys())]
        box_constraints += [net_params[p] <= upper[p] for p in sorted(net_params.keys())]

        # Define primal and dual problems
        if isinstance(self.problem, StochasticPlanningProblem):
            operation_objectives = []
            sd_constraints = []
            primal_constraints = []
            dual_constraints = []

            print(f"Solving stochastic relaxation with {len(self.problem.subproblems)} scenarios.")

            for prob in self.problem.subproblems:
                op, pc, dc, sd = self.setup_inner_problem(prob, net_params, lower, upper)

                operation_objectives += [op]
                primal_constraints += pc
                dual_constraints += dc
                sd_constraints += [sd]

        else:
            operation_objective, primal_constraints, dual_constraints, sd_constraint = (
                self.setup_inner_problem(self.problem, net_params, lower, upper)
            )

            operation_objectives = [operation_objective]
            sd_constraints = [sd_constraint]

        # Create full problem and solve
        problem = cp.Problem(
            cp.Minimize(investment_objective + cp.sum(operation_objectives)),
            box_constraints + sd_constraints + list(primal_constraints) + list(dual_constraints),
        )
        problem.solve(solver=self.solver, **self.solver_kwargs)

        data = {
            "network_parameters": net_params,
            "lower_bounds": lower,
            "upper_bounds": upper,
            "box_constraints": box_constraints,
            "investment_objective": investment_objective,
            "problem": problem,
            "sd_constraint": sd_constraints,
            "primal_constraints": primal_constraints,
            "dual_constraints": dual_constraints,
            "operation_objective": operation_objectives,
        }

        relaxed_parameters = {p: net_params[p].value for p in net_params.keys()}

        return relaxed_parameters, data

    def setup_inner_problem(self, problem, net_params, lower, upper):
        net, devices = problem.layer.network, problem.layer.devices
        dual_devices = zap.dual.dualize(devices, max_price=self.max_price)

        parameters = self.setup_parameters(**net_params)
        envelope_constraints = []
        envelope_variables = []

        primal_costs, primal_constraints, primal_data = net.model_dispatch_problem(
            devices,
            problem.time_horizon,
            dual=False,
            parameters=parameters,
            envelope=(envelope_variables, envelope_constraints),
            lower_param=self.setup_parameters(**lower),
            upper_param=self.setup_parameters(**upper),
        )
        dual_costs, dual_constraints, dual_data = net.model_dispatch_problem(
            dual_devices,
            problem.time_horizon,
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

        operation_objective = problem.operation_objective(y, parameters=parameters, la=cp)

        return operation_objective, primal_constraints, dual_constraints, sd_constraint

    def setup_stochastic_inner_problem():
        # TODO
        raise NotImplementedError
