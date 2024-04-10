from zap.network import DispatchOutcome, PowerNetwork


class AbstractObjective:
    """Abstract implementation of operation objectives."""

    def __call__(self, y: DispatchOutcome, parameters=None):
        return self.forward(y, parameters=parameters)

    def forward(self, y: DispatchOutcome, parameters=None):
        raise NotImplementedError

    @property
    def is_convex(self):
        return False

    @property
    def is_linear(self):
        return False


class DispatchCostObjective(AbstractObjective):
    """Cost of the dispatch outcome."""

    def __init__(self, net: PowerNetwork, devices):
        self.net = net
        self.devices = devices

    def forward(self, y: DispatchOutcome, parameters=None):
        return self.net.operation_cost(
            self.devices, y.power, y.angle, y.local_variables, parameters=parameters
        )

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False


class EmissionsObjective(AbstractObjective):
    """Total emissions of the dispatch outcome."""

    def forward(self, y: DispatchOutcome):
        raise NotImplementedError  # TODO

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return True
