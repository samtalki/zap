from dataclasses import dataclass


class AbstractDevice:
    @property
    def get_terminals(self):
        raise NotImplementedError

    @property
    def num_terminals_per_device(self):
        # TODO - Use get terminals
        raise NotImplementedError

    @property
    def num_devices(self):
        # TODO - Use get terminals
        raise NotImplementedError

    @property
    def is_ac(self):
        return False

    @property
    def is_convex(self):
        return True

    def model_local_variable(self):
        raise NotImplementedError

    def model_cost(self, power, angle, local_variable):
        raise NotImplementedError

    def model_local_constraints(self, power, angle, local_variable):
        raise NotImplementedError


@dataclass
class Injector(AbstractDevice):
    """An Injector is a single-node device that may deposit or withdraw power from the
    network."""

    terminal: int  # TODO
    power_min: int  # TODO
    power_max: int  # TODO
    linear_cost: int  # TODO
    quadratic_cost: int  # TODO


@dataclass
class Transporter(AbstractDevice):
    """A Transport is a two-node device that carries power between nodes. The net power
    of a transporter is always zero."""

    source_terminal: int  # TODO
    sink_terminal: int
    power_min: int
    power_max: int
    linear_cost: int
    quadratic_cost: int
