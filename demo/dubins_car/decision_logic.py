import copy
from enum import Enum, auto


class CarMode(Enum):
    NORMAL = auto()


class State:
    x: float
    y: float
    thetha: float
    v: float
    omega: float

    car_mode: CarMode


def decisionLogic(ego: State, track_map):
    """NOTE: This function is treated very specially by the verse library. I.e., a custom parser is used that might yield different behavior as normal Python code.

    See its documentation on [Agent](https://autoverse-ai.github.io/Verse-library/creating_scenario_in_verse.html#create-an-agent) and [Parser](https://autoverse-ai.github.io/Verse-library/parser.html)."""
    output = copy.deepcopy(ego)
    return output
