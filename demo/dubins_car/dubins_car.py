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
from decision_logic import CarMode  # noqa: E402
from trajectory_class import Trajectory


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
        sensing_error_std: list[float] = [0] * 5,
        traj: Trajectory = Trajectory(
            [("circle", 10, (1, 0), (-1, 0), (0, 0), "counterclockwise")]
        ),
    ):
        super().__init__(id, code, file_name, initial_state, initial_mode)
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_omega = max_omega
        self.max_alpha = max_alpha
        self.control_period = control_period
        self.sensing_error_std = sensing_error_std
        self.traj = traj

    def sensor(self, state):
        error = np.random.normal(0, self.sensing_error_std)
        return (1 + error) * state

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
        t_eval = np.arange(0, time_horizon + time_step, time_step)
        n_points = t_eval.shape[0]

        # All theoretical control instants.
        # NOTE: control_instants SHOULD include the last poing in t_eval due to the way they are iterated in the for-loop below.
        control_instants = np.arange(
            0, t_eval[-1] + self.control_period / 2, self.control_period
        )

        # Find indecies of timestamps where the practical calculations of control inputs should occur
        control_indices = np.searchsorted(t_eval, control_instants, "left")

        trace = np.zeros((n_points, len(initial_set) + 1))
        trace[:, 0] = t_eval

        # Each control period is solved as a separate IVP problem with the states from last period as the initial value, and control input calculated from this initial value
        state = initial_set
        self.traj.reset()
        for i_start, i_end in zip(control_indices[:-1], control_indices[1:]):
            pr, vr = self.traj.get_state(t_eval[i_start])
            u = CarAgent.tracking_controller(
                t_eval[i_start], self.sensor(state), pr, vr
            )
            y = solve_ivp(
                self.dynamics,
                (t_eval[i_start], t_eval[i_end]),
                state,
                t_eval=t_eval[i_start : i_end + 1],
                args=(u,),
            ).y
            trace[i_start : i_end + 1, 1:] = y.T
            state = y[:, -1]

        return trace

    @staticmethod
    def tracking_controller(
        t: float,
        state: list[float] | np.ndarray,
        reference_posture: list[float] | np.ndarray,
        reference_velocity: list[float] | np.ndarray,
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
            reference_velocity: Reference velocities [v_r, omega_r].
                Defaults to [1, 2π/10].
        Returns:
            v_t: Target linear velocity
            omega_t: Target angular velocity
        """
        assert len(state) == 5
        assert len(reference_posture) == 3
        assert len(reference_velocity) == 2

        x, y, theta = state[:3]
        x_r, y_r, theta_r = reference_posture
        v_r, omega_r = reference_velocity
        x_e = np.cos(theta) * (x_r - x) + np.sin(theta) * (y_r - y)
        y_e = -np.sin(theta) * (x_r - x) + np.cos(theta) * (y_r - y)
        theta_e = theta_r - theta

        Kx = 10  # 1/sec
        Ky = 64  # 1/m^2
        Ktheta = 16  # 1/m

        v_t = v_r * np.cos(theta_e) + Kx * x_e
        omega_t = omega_r + v_r * (Ky * y_e + Ktheta * np.sin(theta_e))

        return v_t, omega_t

    def reference_trace(self, t_eval: list[float]) -> np.ndarray:
        self.traj.reset()
        return np.asarray(
            [
                [t, *posture, *velocity]
                for t in t_eval
                for posture, velocity in [self.traj.get_state(t)]
            ]
        )

    def plot_reference_trace(self, fig: go.Figure, n_points: int = 50):
        t_eval = np.linspace(0, self.traj.total_duration, n_points)
        reference_trace = self.reference_trace(t_eval)

        # Plot at most max_n_points to avoid overcrowding
        def plot_row(row):
            t, x, y, theta, v, omega = row
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
                    showlegend=False,
                    name="Reference",
                    hovertext=f"t={t:.2f}, x={x:.2f}, y={y:.2f}",
                    hoverinfo="text",
                )
            )

        np.apply_along_axis(plot_row, axis=1, arr=reference_trace)
        return fig


if __name__ == "__main__":
    # >>>>>>>>> Simulation Parameters >>>>>>>>>
    time_step = 0.001

    # center and error for initial_set =[x, y, theta, v, omega]
    initial_set_c = (1.0, 0.0, np.pi / 2, 0.0, 0.0)
    initial_set_e = (0.1,) * 5

    sensing_errors = [0.0] * 2 + [0, 0, 0]

    traj = Trajectory(
        [
            ("circle", 10, (1, 0), (-1, 0), (0, 0), "counterclockwise"),
            ("line", 5, (-1, 0), (0, 0)),
            ("circle", 20, (0, 0), (2, 0), (1, 0), "counterclockwise"),
            # ("circle", 5, (2, 0), (1, 0), (1.5, 0), "counterclockwise")
        ]
    )
    time_horizon = traj.total_duration
    # <<<<<<<<< Simulation Parameters <<<<<<<<<

    # File containing decision logic
    CAR_DL = os.path.join(os.path.dirname(__file__), "decision_logic.py")

    car1 = CarAgent(
        "car1",
        file_name=CAR_DL,
        # Sensing errors only in x and y
        sensing_error_std=sensing_errors,
        traj=traj,
    )
    dubins_car = Scenario(ScenarioConfig(parallel=False, print_level=0))
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

    traces = dubins_car.simulate_multi(time_horizon, time_step, max_height=6, n_sims=1)

    fig = go.Figure()
    # simulation_tree(traces[0], None, fig, 1, 2, [0, 1, 2], "fill", "trace")
    for trace in traces:
        simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    car1.plot_reference_trace(fig, n_points=70)
    fig.show()
