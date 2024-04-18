import unittest
import cvxpy as cp
import numpy as np
import pandas as pd

import zap
from zap.tests import network_examples as examples


KKT_TOLERANCE = 1e-2
JACOBIAN_PERTURBATION_RANGE = [1.0, 1e-1, 1e-2, 1e-3]
JACOBIAN_ABSOLUTE_TOLERANCE = 1e-7
JACOBIAN_RELATIVE_TOLERANCE = 1e-4

# TODO Resolve this warning
pd.set_option("future.no_silent_downcasting", True)


class TestDualSimple(unittest.TestCase):
    @classmethod
    def load_network_data(cls):
        net, devices, param = examples.load_test_network()
        devices[2].linear_cost *= 0.0  # No transmission line costs
        return net, devices, param

    @classmethod
    def setUpClass(cls):
        cls.net, cls.devices, cls.parameters = cls.load_network_data()

        # Add ground
        cls.devices += [
            zap.Ground(num_nodes=cls.net.num_nodes, terminal=np.array([0]), voltage=np.array([0.0]))
        ]

        # Dualize
        cls.dual_devices = zap.dual.dualize(cls.devices)

        cls.primal = cls.net.dispatch(
            cls.devices,
            add_ground=False,
            solver=cp.MOSEK,
        )
        cls.dual = cls.net.dispatch(
            cls.dual_devices,
            add_ground=False,
            solver=cp.MOSEK,
            dual=True,
        )

    def test_objective_value(self):
        np.testing.assert_allclose(
            -self.dual.problem.value, self.primal.problem.value, atol=1e-3, rtol=1e-6
        )

    def test_prices(self, atol=1e-2, rtol=1e-2):
        self.assertLessEqual(
            np.linalg.norm(self.dual.global_angle - self.primal.prices),
            atol + rtol * np.linalg.norm(self.primal.prices),
        )
        self.assertLessEqual(
            np.linalg.norm(self.dual.prices - self.primal.global_angle),
            atol + rtol * np.linalg.norm(self.primal.global_angle),
        )


class TestDualPyPSA(TestDualSimple):
    @classmethod
    def load_network_data(cls):
        return examples.load_pypsa24hour()
