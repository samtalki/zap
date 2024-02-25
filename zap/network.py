from dataclasses import dataclass


@dataclass
class PowerNetwork:
    """A power network defines the domain (nodes and settlement points) of the
    electrical system."""

    num_nodes: int
