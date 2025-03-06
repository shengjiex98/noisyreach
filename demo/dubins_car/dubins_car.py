import os
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from verse import BaseAgent, Scenario, ScenarioConfig
from verse.map import LaneMap
from verse.plotter.plotter2D import go, simulation_tree

sys.path.append(str(Path(__file__).parent))
print(sys.path)
from decision_logic import CarMode  # noqa: E402


class CarAgent(BaseAgent):
    def __init__(
        self,
        id,
        code=None,
        file_name=None,
        initial_state=None,
        initial_mode=None,
        # Numeric values taken from [1] Y. Kanayama, Y. Kimura, F. Miyazaki, and T. Noguchi, “A stable tracking control method for an autonomous mobile robot,” in Proceedings., IEEE International Conference on Robotics and Automation, Cincinnati, OH, USA: IEEE Comput. Soc. Press, 1990, pp. 384–389. doi: 10.1109/ROBOT.1990.126006.
        max_speed: float = 0.4,
        max_accel: float = 0.5,
        max_omega: float = 0.8,
        max_alpha: float = 5,
        control_period: float = 0.02,
    ):
        super().__init__(id, code, file_name, initial_state, initial_mode)
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_omega = max_omega
        self.max_alpha = max_alpha
        self.control_period = control_period

    def dynamics(self, t, state, u):
        assert len(state) == 5
        assert len(u) == 2
        x, y, theta, v, omega = state
        v_t, omega_t = u

        v_t = min(self.max_speed, max(0, v_t))
        omega_t = min(self.max_omega, max(-self.max_omega, omega_t))

        x_dot = v_t * np.cos(theta)
        y_dot = v_t * np.sin(theta)
        theta_dot = omega_t
        v_dot = min(self.max_accel, max(-self.max_accel, v_t - v))
        omega_dot = min(self.max_alpha, max(-self.max_alpha, omega_t))

        return [x_dot, y_dot, theta_dot, v_dot, omega_dot]

    @staticmethod
    def reference_posture(t) -> list[float]:
        theta_r = np.pi / 10 * t + np.pi / 2
        return [np.cos(theta_r - np.pi / 2), np.sin(theta_r - np.pi / 2), theta_r]

    @staticmethod
    def reference_velocities(t) -> list[float]:
        return [np.pi / 10, np.pi / 10]

    @staticmethod
    def reference_trace(time_horizon: float, time_step: float) -> np.ndarray:
        return np.asarray(
            [
                [t] + CarAgent.reference_posture(t) + CarAgent.reference_velocities(t)
                for t in np.arange(0, time_horizon + time_step, time_step)
            ]
        )

    @staticmethod
    def tracking_controller(
        t: float,
        state: list[float] | np.ndarray,
        reference_posture=None,
        reference_velocities=None,
    ) -> tuple[float, float]:
        """
        Implements a tracking controller for a Dubins car model.
        This controller generates control inputs (linear and angular velocities) to track a reference trajectory.
        If no reference trajectory is provided, it defaults to tracking a circular path.

        Adapted from Y. Kanayama, Y. Kimura, F. Miyazaki, and T. Noguchi, “A stable tracking control method for an autonomous mobile robot,” in Proceedings., IEEE International Conference on Robotics and Automation, 1990. doi: 10.1109/ROBOT.1990.126006.
        Args:
            t: Current time
            state: Current state vector [x, y, theta, v, omega]
            reference_posture: Reference posture [x_r, y_r, theta_r].
                Defaults to circular trajectory.
            reference_velocities: Reference velocities [v_r, omega_r].
                Defaults to [1, 2π/10].
        Returns:
            v_t: Target linear velocity
            omega_t: Target angular velocity
        """
        if not reference_posture:
            reference_posture = CarAgent.reference_posture(t)
        if not reference_velocities:
            reference_velocities = CarAgent.reference_velocities(t)

        assert len(state) == 5
        assert len(reference_posture) == 3
        assert len(reference_velocities) == 2

        x, y, theta = state[:3]
        x_r, y_r, theta_r, v_r, omega_r = reference_posture + reference_velocities
        x_e = np.cos(theta) * (x_r - x) + np.sin(theta) * (y_r - y)
        y_e = -np.sin(theta) * (x_r - x) + np.cos(theta) * (y_r - y)
        theta_e = theta_r - theta

        Kx = 10  # 1/sec
        Ky = 64  # 1/m^2
        Ktheta = 16  # 1/m

        v_t = v_r * np.cos(theta_e) + Kx * x_e
        omega_t = omega_r + v_r * (Ky * y_e + Ktheta * np.sin(theta_e))

        return v_t, omega_t

    def TC_simulate(
        self,
        mode: str,
        initial_set: list[float],
        time_horizon: float,
        time_step: float,
        map: LaneMap = None,
    ):
        # TODO: currently only using Decimal for assert, not integration. Consider using Decimals also for calculation, or other ways to help with precision/rounding errors.
        assert Decimal(str(self.control_period)) % Decimal(str(time_step)) == 0, (
            f"control_period must be multiples of simulation's time_step. Got {self.control_period, time_step}"
        )

        # All timestamps to simulate
        t = np.arange(0, time_horizon + time_step, time_step)
        n_points = t.shape[0]

        # All theoretical control instants.
        control_instants = np.arange(
            0, t[-1] + self.control_period, self.control_period
        )

        # Find indecies of timestamps where the practical calculations of control inputs should occur
        control_indices = np.searchsorted(t, control_instants, "left")

        trace = np.zeros((n_points, len(initial_set) + 1))
        trace[:, 0] = t

        # Each control period is solved as a separate IVP problem with the states from last period as the initial value, and control input calculated from this initial value
        state = initial_set
        for i_start, i_end in zip(control_indices[:-1], control_indices[1:]):
            u = CarAgent.tracking_controller(t[i_start], state)
            y = solve_ivp(
                self.dynamics,
                (t[i_start], t[i_end]),
                state,
                t_eval=t[i_start : i_end + 1],
                args=(u,),
            ).y
            trace[i_start : i_end + 1, 1:] = y.T
            state = y[:, -1]

        return trace

    @staticmethod
    def plot_reference_trace(
        fig: go.Figure, time_horizon: float, time_step: float, max_n_points: int = 50
    ):
        reference_trace = CarAgent.reference_trace(time_horizon, time_step)
        n = reference_trace.shape[0]

        # Plot at most max_n_points to avoid overcrowding
        for i in range(0, n, (n - 1) // (max_n_points - 1)):
            t, x, y, theta, v, omega = reference_trace[i, :]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(
                        size=20 * v,
                        symbol="arrow",
                        angle=-theta / np.pi * 180 + 90,
                        color="green",
                        line=dict(width=2, color="DarkSlateGrey"),
                    ),
                    showlegend=True if i == 0 else False,
                    name="Reference",
                    hovertext=f"t={t:.2f}, x={x:.2f}, y={y:.2f}",
                    hoverinfo="text",
                )
            )
        return fig


if __name__ == "__main__":
    # >>>>>>>>> Simulation Parameters >>>>>>>>>
    time_horizon = 9.8
    time_step = 0.001

    # center and error for initial_set =[x, y, theta, v, omega]
    initial_set_c = (1.0, 0.0, np.pi / 2, 0.0, 0.0)
    initial_set_e = (0.1,) * 5
    # <<<<<<<<< Simulation Parameters <<<<<<<<<

    # File containing decision logic
    CAR_DL = os.path.join(os.path.dirname(__file__), "decision_logic.py")

    car1 = CarAgent("car1", file_name=CAR_DL)
    dubins_car = Scenario(ScenarioConfig(parallel=False))
    dubins_car.add_agent(car1)
    dubins_car.set_init(
        # Continuous states
        [
            [
                [c - e for (c, e) in zip(initial_set_c, initial_set_e)],
                [c + e for (c, e) in zip(initial_set_c, initial_set_e)],
            ]
        ],
        # Discrete states (modes)
        [(CarMode.NORMAL,)],
    )

    traces = dubins_car.simulate_multi(time_horizon, time_step, max_height=6)

    fig = go.Figure()
    # simulation_tree(traces[0], None, fig, 1, 2, [0, 1, 2], "fill", "trace")
    for trace in traces:
        simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    CarAgent.plot_reference_trace(fig, time_horizon, time_step)
    fig.show()
