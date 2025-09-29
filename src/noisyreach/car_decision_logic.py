"""Decision logic for car agents in noisy reachability analysis.

This module defines the operational modes and state representation for car agents
used in the VERSE simulation framework. The decision logic function is processed
by VERSE's custom parser for agent behavior specification.
"""

import copy
from dataclasses import dataclass
from enum import Enum, auto


class CarMode(Enum):
    """Operating modes for car agents.

    This enumeration defines the discrete operational states that a car agent
    can be in during simulation. Currently supports normal operation mode.

    Attributes:
        NORMAL: Standard operation mode for trajectory following
    """

    NORMAL = auto()


@dataclass
class State:
    """State representation for car agents.

    Represents the complete state of a car agent including position,
    orientation, velocities, and operational mode.

    Attributes:
        x: X-coordinate position in meters
        y: Y-coordinate position in meters
        theta: Heading angle in radians (0 = east, π/2 = north)
        v: Linear velocity in m/s
        omega: Angular velocity in rad/s
        car_mode: Current operational mode of the agent
    """

    x: float
    y: float
    theta: float
    v: float
    omega: float

    car_mode: CarMode


def decisionLogic(ego: State, track_map) -> State:
    """Decision logic function for car agents.

    This function defines the discrete decision-making logic for car agents.
    Currently implements a simple pass-through behavior where the agent
    maintains its current state and mode.

    IMPORTANT: This function is processed by VERSE's custom parser, which may
    behave differently from standard Python execution. The parser analyzes the
    function to extract discrete transition logic for formal verification.

    Args:
        ego: Current state of the car agent
        track_map: Map information for decision making (currently unused)

    Returns:
        Updated state after applying decision logic

    Note:
        See VERSE documentation for details on the special parsing behavior:
        - Agent: https://autoverse-ai.github.io/Verse-library/creating_scenario_in_verse.html#create-an-agent
        - Parser: https://autoverse-ai.github.io/Verse-library/parser.html
    """
    output = copy.deepcopy(ego)
    return output
