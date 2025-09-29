"""Car agent implementation for noisy reachability analysis.

This module provides CarAgent, an autonomous car simulation agent that extends
VERSE's BaseAgent with trajectory tracking capabilities, sensing noise modeling,
and Dubins car dynamics.

The agent implements a tracking controller based on Kanayama et al.'s stable
tracking method, with configurable sensing errors and control periods for
reachability analysis under uncertainty.

Reference:
    Y. Kanayama, Y. Kimura, F. Miyazaki, and T. Noguchi, "A stable tracking
    control method for an autonomous mobile robot," IEEE ICRA, 1990.
"""

from decimal import Decimal
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from verse import BaseAgent
from verse.map import LaneMap

from noisyreach.car_decision_logic import CarMode as CarMode
from noisyreach.trajectory import Trajectory

CAR_DL = Path(__file__).parent.joinpath("car_decision_logic.py")


class CarAgent(BaseAgent):
    """Car agent with trajectory tracking and sensing noise.

    Extends VERSE BaseAgent to provide a realistic car model with:
    - Dubins car dynamics (position, orientation, velocities)
    - Trajectory tracking controller with configurable gains
    - Sensing noise modeling for reachability analysis
    - Configurable physical constraints (speed, acceleration limits)

    The agent follows reference trajectories while accounting for sensing
    errors and control delays, making it suitable for safety analysis.
    """

    def __init__(
        self,
        id,
        code=None,
        file_name=CAR_DL,
        initial_state=None,
        initial_mode=None,
        seed=None,
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
        """Initialize car agent with physical and control parameters.

        Args:
            id: Unique identifier for this agent
            code: Custom decision logic code (optional)
            file_name: Path to decision logic file (default: car_decision_logic.py)
            initial_state: Initial state vector [x, y, theta, v, omega]
            initial_mode: Initial discrete mode (default: CarMode.NORMAL)
            seed: Random seed for deterministic sensing noise
            max_speed: Maximum linear velocity in m/s
            max_accel: Maximum linear acceleration in m/s²
            max_omega: Maximum angular velocity in rad/s
            max_alpha: Maximum angular acceleration in rad/s²
            control_period: Control update period in seconds
            sensing_error_std: Standard deviations for sensing noise (length 5)
            traj: Reference trajectory to follow

        Note:
            Physical parameters are based on Kanayama et al. (1990) for stable
            tracking control of autonomous mobile robots.
        """
        super().__init__(id, code, file_name, initial_state, initial_mode)
        self.seed = seed
        self.nonce = 0
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_omega = max_omega
        self.max_alpha = max_alpha
        self.control_period = control_period
        self.sensing_error_std = sensing_error_std
        self.traj = traj

    def sensor(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply sensing noise to state measurements.

        Models multiplicative Gaussian sensing errors affecting each state component.
        The noise model is: sensed_state = (1 + noise) * true_state where
        noise ~ N(0, sensing_error_std).

        Args:
            state: True system state [x, y, theta, v, omega]
            rng: Random number generator for noise sampling

        Returns:
            Noisy state measurement with same dimensions as input
        """
        error = rng.normal(0, self.sensing_error_std)
        return (1 + error) * state

    def dynamics(
        self, t: float, state: list[float], u: tuple[float, float]
    ) -> list[float]:
        """Compute system dynamics for Dubins car model.

        Implements the continuous-time dynamics of a car-like vehicle with
        velocity and angular velocity as control inputs. Physical constraints
        are applied to maintain realistic motion.

        Args:
            t: Current time (unused for time-invariant system)
            state: Current state [x, y, theta, v, omega] where:
                - x, y: position coordinates in meters
                - theta: heading angle in radians
                - v: linear velocity in m/s
                - omega: angular velocity in rad/s
            u: Control inputs (v_target, omega_target)

        Returns:
            State derivatives [x_dot, y_dot, theta_dot, v_dot, omega_dot]

        Note:
            Control inputs are saturated to respect physical limits for
            speed, angular velocity, acceleration, and angular acceleration.
        """
        assert len(state) == 5
        assert len(u) == 2
        x, y, theta, v, omega = state
        v_t, omega_t = u

        # Apply velocity and angular velocity constraints
        v_t = min(self.max_speed, max(0, v_t))
        omega_t = min(self.max_omega, max(-self.max_omega, omega_t))

        # Kinematic equations for car-like vehicle
        x_dot = v_t * np.cos(theta)
        y_dot = v_t * np.sin(theta)
        theta_dot = omega_t

        # Apply acceleration constraints
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
    ) -> np.ndarray:
        """Simulate trajectory tracking with control periods.

        Performs closed-loop simulation of the car agent following its reference
        trajectory with discrete control updates and sensing noise. The simulation
        accounts for sensor-to-actuator delay by computing control inputs based
        on measurements from the previous control period.

        Args:
            mode: Simulation mode (currently unused)
            initial_set: Initial state vector [x, y, theta, v, omega]
            time_horizon: Total simulation time in seconds
            time_step: Simulation integration time step in seconds
            map: Lane map for decision logic (optional, currently unused)

        Returns:
            Simulation trace array (n_points, 6) where:
            - Column 0: time stamps
            - Columns 1-5: state variables [x, y, theta, v, omega]

        Warning:
            Issues warning if control_period is not a multiple of time_step
        """
        # TODO: currently only using Decimal for assert, not integration. Consider using Decimals also for calculation, or other ways to help with precision/rounding errors.
        if Decimal(str(self.control_period)) % Decimal(str(time_step)) != 0:
            print(
                f"Warning: control_period must be multiples of simulation's time_step. Got {self.control_period, time_step}"
            )

        # Set seed for deterministic sampling in `self.sensor()`. Incrementing `self.nonce` each time ensures different sampling when running `TC_simulate()` multiple times.
        if self.seed:
            rng = np.random.default_rng(self.seed + self.nonce)
            self.nonce += 1
        else:
            rng = np.random.default_rng()

        # All timestamps to simulate
        t_eval = np.arange(0, time_horizon + time_step, time_step)
        n_points = t_eval.shape[0]

        # All theoretical control instants.
        # NOTE: control_instants SHOULD include the last point in t_eval due to the way they are iterated in the for-loop below.
        control_instants = np.append(
            np.arange(0, t_eval[-1], self.control_period), [t_eval[-1]]
        )

        # Find indices of timestamps where the practical calculations of control inputs should occur
        control_indices = np.searchsorted(t_eval, control_instants, "left")

        trace = np.zeros((n_points, len(initial_set) + 1))
        trace[:, 0] = t_eval

        # Each control period is solved as a separate IVP problem with the states from last period as the initial value, and control input calculated from this initial value
        state = initial_set
        u = (0, 0)
        self.traj.reset()
        for i_start, i_end in zip(control_indices[:-1], control_indices[1:]):
            pr, vr = self.traj.get_state(t_eval[i_start])
            u_next = CarAgent.tracking_controller(
                t_eval[i_start], self.sensor(state, rng), pr, vr
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
            u = u_next

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
        """Generate reference trajectory trace for given time points.

        Computes the reference trajectory states at specified time points,
        useful for comparison with actual agent trajectory or visualization.

        Args:
            t_eval: List of time points to evaluate

        Returns:
            Array (n_points, 6) containing [t, x, y, theta, v, omega] for each time point
        """
        self.traj.reset()
        return np.asarray(
            [
                [t, *posture, *velocity]
                for t in t_eval
                for posture, velocity in [self.traj.get_state(t)]
            ]
        )

    def plot_reference_trace(self, fig: go.Figure, n_points: int = 50) -> go.Figure:
        """Add reference trajectory visualization to Plotly figure.

        Plots the reference trajectory as arrows showing position and orientation
        at evenly spaced time points. Arrow size represents velocity magnitude.

        Args:
            fig: Plotly figure to add traces to
            n_points: Number of trajectory points to visualize (default: 50)

        Returns:
            Updated Plotly figure with reference trajectory traces
        """
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
