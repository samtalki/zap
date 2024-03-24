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

        # Check dimensions
        time_horizon = self.time_horizon
        num_devices = len(self.devices) + 1  # Includes ground
        num_nodes = self.net.num_nodes
        result = self.dispatch

        # Each variable should have the same number of elements as the number of devices
        for y in [
            result.power,
            result.angle,
            result.local_variables,
            result.phase_duals,
            result.local_equality_duals,
            result.local_inequality_duals,
        ]:
            self.assertEqual(len(y), num_devices)

        # Power and angle should each be (num_terminals, num_devices, time_horizon)
        for y in [result.power, result.angle]:
            for y_dev, dev in zip(y, self.devices + [result.ground]):
                if y_dev is not None:
                    for y_dev_term in y_dev:
                        self.assertEqual(y_dev_term.shape, (dev.num_devices, time_horizon))

        # Global variables should be (num_nodes, time_horizon)
        self.assertEqual(result.global_angle.shape, (num_nodes, time_horizon))
        self.assertEqual(result.prices.shape, (num_nodes, time_horizon))

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
