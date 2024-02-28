import pypsa
import numpy as np
import pandas as pd

from zap.network import PowerNetwork
from zap.devices.injector import Generator, Load
from zap.devices.transporter import DCLine, ACLine
from zap.devices.store import Battery


def parse_buses(net: pypsa.Network):
    buses = net.buses.loc[net.buses.carrier != "battery"].index
    buses_to_index = {bus: i for i, bus in enumerate(buses)}
    return buses, buses_to_index


def parse_network(net: pypsa.Network):
    buses, buses_to_index = parse_buses(net)
    return PowerNetwork(len(buses))


def build_dynamic(device, device_t, key, dates):
    # Load dynamic data
    dyn = device_t[key].loc[dates].copy()
    dyn_index = [device.index.get_loc(c) for c in dyn.columns]

    # Replace static values with their dynamic counterparts
    dynamic_values = np.outer(device[key].values, np.ones(len(dates)))
    dynamic_values[dyn_index, :] = dyn.values.T
    return dynamic_values


def parse_generators(net: pypsa.Network, dates):
    buses, buses_to_index = parse_buses(net)
    terminals = net.generators.bus.replace(buses_to_index).values.astype(int)

    # Build dynamic capacities
    dynamic_capacities = build_dynamic(
        net.generators, net.generators_t, "p_max_pu", dates
    )
    dynamic_costs = build_dynamic(
        net.generators, net.generators_t, "marginal_cost", dates
    )

    return Generator(
        num_nodes=len(buses),
        terminal=terminals,
        nominal_capacity=net.generators.p_nom.values,
        dynamic_capacity=dynamic_capacities,
        linear_cost=dynamic_costs,
    )


def parse_loads(net: pypsa.Network, dates, marginal_load_value=10_000.0):
    buses, buses_to_index = parse_buses(net)
    terminals = net.loads.bus.replace(buses_to_index).values.astype(int)
    load = build_dynamic(net.loads, net.loads_t, "p_set", dates)

    return Load(
        num_nodes=len(buses),
        terminal=terminals,
        load=load,
        linear_cost=marginal_load_value * np.ones(len(buses)),
    )


def get_source_sinks(df: pd.DataFrame, buses_to_index):
    sources = df.bus0.replace(buses_to_index).values.astype(int)
    sinks = df.bus1.replace(buses_to_index).values.astype(int)

    return sources, sinks


def parse_dc_lines(net: pypsa.Network):
    buses, buses_to_index = parse_buses(net)

    links = net.links[net.links.carrier == "DC"]
    sources, sinks = get_source_sinks(links, buses_to_index)

    return DCLine(
        num_nodes=len(buses),
        source_terminal=sources,
        sink_terminal=sinks,
        capacity=links.p_max_pu.values,
        nominal_capacity=links.p_nom.values,
    )


def parse_ac_lines(net: pypsa.Network):
    buses, buses_to_index = parse_buses(net)

    sources, sinks = get_source_sinks(net.lines, buses_to_index)

    # Compute per-MW susceptance
    susceptance = 1 / net.lines.x.values
    susceptance = np.divide(susceptance, net.lines.s_nom.values)

    return ACLine(
        num_nodes=len(buses),
        source_terminal=sources,
        sink_terminal=sinks,
        susceptance=susceptance,
        capacity=net.lines.s_max_pu.values,
        nominal_capacity=net.lines.s_nom.values,
    )


def parse_batteries(net: pypsa.Network):
    buses, buses_to_index = parse_buses(net)
    terminals = net.storage_units.bus.replace(buses_to_index).values.astype(int)

    return Battery(
        num_nodes=len(buses),
        terminal=terminals,
        power_capacity=net.storage_units.p_nom.values,
        duration=net.storage_units.max_hours.values,
        charge_efficiency=net.storage_units.efficiency_dispatch.values,
    )


def load_pypsa_network(net: pypsa.Network, dates):
    network = PowerNetwork(len(net.buses))

    devices = [
        parse_generators(net, dates),
        parse_loads(net, dates),
        parse_dc_lines(net),
        parse_ac_lines(net),
        parse_batteries(net),
    ]

    return network, devices
