import unittest
import cvxpy as cp
import numpy as np

from zap.tests import network_examples as examples


KKT_TOLERANCE = 1e-2


class BaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net, cls.devices, cls.parameters = cls.load_network_data()
        cls.time_horizon = cls.devices[0].time_horizon
        cls.dispatch = cls.net.dispatch(
            cls.devices, time_horizon=cls.time_horizon, parameters=cls.parameters, solver=cp.MOSEK
        )

    @classmethod
    def load_network_data(cls):
        return examples.load_simple_network()

    def test_network_solve(self):
        # Check solution status
        self.assertTrue(self.dispatch.problem.status == "optimal")

        # TODO Check dimensions
        pass

    def test_kkt(self):
        K = self.net.kkt(self.devices, self.dispatch, parameters=self.parameters)

        # Check dimensions
        for s1, s2 in zip(K.shape, self.dispatch.shape):
            self.assertEqual(s1, s2)

        # Check small norm
        K_vec = K.vectorize()
        self.assertLessEqual(np.linalg.norm(K_vec) / np.sqrt(K_vec.size), KKT_TOLERANCE)

    def test_jacobian(self):
        net, devices, dispatch, parameters = self.net, self.devices, self.dispatch, self.parameters

        K = net.kkt(devices, dispatch, parameters=parameters)
        jac = net.kkt_jacobian_variables(devices, dispatch, parameters=parameters)

        # Check dimensions
        self.assertEqual(jac.shape[0], K.vectorize().size)

        # TODO Check derivatives numerically
        # Should shrink as we decrease the perturbation
        pass


# TODO - Make this programmatic
class TestNetworkDispatchSimple(BaseTest):
    @classmethod
    def load_network_data(cls):
        return examples.load_simple_network()

    # TODO Check specific outcomes


class TestNetworkDispatchPyPSA1Hour(BaseTest):
    @classmethod
    def load_network_data(cls):
        return examples.load_pypsa1hour()


class TestNetworkDispatchPyPSA24Hour(BaseTest):
    @classmethod
    def load_network_data(cls):
        return examples.load_pypsa24hour()
