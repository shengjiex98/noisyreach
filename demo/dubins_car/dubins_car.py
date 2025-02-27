import copy
import os

import numpy as np
from verse import BaseAgent


def decisionLogic(ego, track_map):
    """NOTE: This function is treated very specially by the verse library. I.e., a custom parser is used that might yield different behavior as normal Python code.

    See its documentation on [Agent](https://autoverse-ai.github.io/Verse-library/creating_scenario_in_verse.html#create-an-agent) and [Parser](https://autoverse-ai.github.io/Verse-library/parser.html)."""
    output = copy.deepcopy(ego)
    x, y, theta, v, omega = ego
    if x > 0:
        output[0] = x
    if y > 0:
        output[1] = y
    return output


class CarAgent(BaseAgent):
    def __init__(
        self,
        id,
        code=None,
        file_name=None,
        initial_state=None,
        initial_mode=None,
        speed: float = 2,
        accel: float = 1,
    ):
        super().__init__(id, code, file_name, initial_state, initial_mode)
        self.speed = speed
        self.accel = accel

    @staticmethod
    def dynamics(t, state, u=[0, 0]):
        x, y, theta, v, omega = state
        a, alpha = u
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega
        v_dot = a
        omega_dot = alpha
        return [x_dot, y_dot, theta_dot, v_dot, omega_dot]

    def controller(self):
        return (0, 0)

    # def TC_simulate(
    #     self,
    #     mode: str,
    #     initialSet: list[float],
    #     time_horizon: float,
    #     time_step: float,
    #     map: LaneMap = None,
    # ):
    #     num_points = int(np.ceil(time_horizon / time_step))
    #     trace = np.zeros((num_points + 1, 1 + len(initialSet)))
    #     trace[:, 0] = np.arange(0, time_horizon + time_step, time_step)
    #     trace[0, 1:] = initialSet
    #     state = initialSet
    #     for i in range(num_points):
    #         a, alpha = self.controller(state)
    #         sol = solve_ivp(
    #             self.dynamic,
    #             (state[0], state[0] + time_step),
    #             state,
    #             args=((a, alpha),),
    #         )
    #         print(sol)


if __name__ == "__main__":
    car = CarAgent(
        "car1", file_name=os.path.join(os.path.dirname(__file__), "decision_logic.py")
    )
    # car = CarAgent("car1", file_name=__file__)
    traj = car.TC_simulate("normal", [1, 0, np.pi / 2, 0, np.pi / 10], 5, 0.1)
    print(type(traj), traj, sep="\n")
