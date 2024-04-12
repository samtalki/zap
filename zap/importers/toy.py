import numpy as np

from zap.network import PowerNetwork
from zap.devices import Generator, Load, ACLine, Battery


def load_test_network(num_nodes=7):
    net = PowerNetwork(num_nodes)

    generators = Generator(
        num_nodes=num_nodes,
        terminal=np.array([0, 1, 3]),
        dynamic_capacity=np.array(
            [
                np.ones(4),  # Peaker
                [0.2, 1.0, 1.0, 0.3],  # Solar panel
                np.ones(4),  # CC Gas
            ]
        ),
        linear_cost=np.array([100.0, 0.5, 40.0]),
        nominal_capacity=np.array([100.0, 50.0, 15.0]),
        capital_cost=np.array([4.0, 10.0, 10.0]),
        emission_rates=np.array([800.0, 0.0, 500.0]),
    )

    loads = Load(
        num_nodes=num_nodes,
        terminal=np.array([0]),
        load=np.array([[30.0, 40.0, 45.0, 80.0]]),
        linear_cost=np.array([200.0]),
    )

    links = ACLine(
        num_nodes=num_nodes,
        source_terminal=np.array([0, 1, 3]),
        sink_terminal=np.array([1, 3, 0]),
        capacity=np.ones(3),
        susceptance=np.array([0.1, 0.05, 1.0]),
        nominal_capacity=np.array([45.0, 50.0, 11.0]),
        linear_cost=0.025 * np.ones(3),
        capital_cost=np.array([100.0, 25.0, 30.0]),
    )

    batteries = Battery(
        num_nodes=num_nodes,
        terminal=np.array([1]),
        power_capacity=np.array([5.0]),
        duration=np.array([4.0]),
        linear_cost=np.array([0.01]),
    )

    devices = [generators, loads, links, batteries]
    return net, devices
