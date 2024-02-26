from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray


class AbstractDevice(ABC):
    # Pre-defined methods

    @property
    def num_terminals_per_device(self) -> int:
        terminals = self.get_terminals
        assert len(terminals.shape) <= 2

        if len(terminals.shape) == 1:
            return 1
        else:
            return terminals.shape[1]

    @property
    def num_devices(self):
        return self.get_terminals.shape[0]

    # Optionally defined methods

    @property
    def is_ac(self):
        return False

    @property
    def is_convex(self):
        return True

    @property
    def data(self):
        pass

    def model_local_variables(self):
        return None

    # Sub classes must define these methods

    @property
    @abstractmethod
    def get_terminals(self) -> NDArray:
        pass

    @abstractmethod
    def model_cost(self, power, angle, local_variable):
        raise NotImplementedError

    @abstractmethod
    def model_local_constraints(self, power, angle, local_variable):
        raise NotImplementedError


@dataclass(kw_only=True)
class Transporter(AbstractDevice):
    """A Transport is a two-node device that carries power between nodes. The net power
    of a transporter is always zero."""

    source_terminal: NDArray
    sink_terminal: NDArray
    power_min: NDArray
    power_max: NDArray
    linear_cost: NDArray
    quadratic_cost: Optional[NDArray] = None

    def __post_init__(self):
        # TODO - Add dimension checks
        pass
