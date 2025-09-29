"""Deviation analysis for noisy reachability problems.

This module provides functions to analyze the maximum deviation of autonomous
agents from their reference trajectories under sensing noise and control delays.
The analysis is performed through Monte Carlo simulation using the VERSE framework.

The main functionality includes:
- deviation(): Core function for computing trajectory deviations
- trace_deviation(): Extract deviation metrics from simulation traces
- get_max_diam(): Compute maximum diameter of reachable sets

Example:
    Analyze car agent deviation with 2% latency and 90% sensing accuracy:

    >>> from noisyreach import deviation
    >>> deviations = deviation(latency=0.02, accuracy=0.9, system="CAR")
    >>> max_dev = np.max(deviations)
    >>> print(f"Maximum deviation: {max_dev:.3f} meters")
"""

import numpy as np
import plotly.graph_objects as go
from verse import Scenario, ScenarioConfig
from verse.analysis import AnalysisTree
from verse.plotter.plotter2D import simulation_tree

from .car_agent import CarAgent, CarMode
from .trajectory import Trajectory

AVAIL_SYSTEMS = {"CAR": {"dims": 5, "desc": "Dimensions are: (x, y, theta, v, omega)."}}
DIMS = {"CAR": 5}


def get_max_diam(latency: float, errors: float | list[float], sysname: str) -> float:
    """Compute maximum diameter of reachable set.

    Calculates the maximum diameter of the reachable set for given latency
    and sensing errors by converting error rates to accuracy and running
    deviation analysis.

    Args:
        latency: Control latency in seconds
        errors: Sensing error rates (scalar or list)
        sysname: System name (currently supports "CAR")

    Returns:
        Maximum diameter of reachable set in meters

    Note:
        This function converts error rates to accuracy rates (1 - error)
        before calling the main deviation analysis.
    """
    if isinstance(errors, float):
        errors = [errors] * 5
    return np.max(deviation(latency, [1 - e for e in errors], system=sysname))


def trace_deviation(traces: list[AnalysisTree], agent: CarAgent) -> np.ndarray:
    """Extract position deviations from simulation traces.

    Computes the Euclidean distance between actual agent trajectories and
    the reference trajectory for each simulation run.

    Args:
        traces: List of simulation traces from VERSE analysis
        agent: Car agent containing the reference trajectory

    Returns:
        Array of maximum deviations (n_simulations,) where each element
        is the maximum Euclidean distance from reference for that simulation

    Note:
        Only considers x,y position deviations, ignoring orientation and velocities.
    """
    traces = np.asarray([trace.root.trace[agent.id] for trace in traces])
    reference_trace = agent.reference_trace(list(traces[0, :, 0]))

    diff = traces[:, :, 1:3] - np.expand_dims(reference_trace, 0)[:, :, 1:3]

    return np.max(
        np.apply_along_axis(lambda row: np.hypot(row[0], row[1]), -1, diff),
        axis=1,
        keepdims=False,
    )


def deviation(
    latency: float,
    accuracy: float | list[float],
    system: str = "CAR",
    num_sims: int = 10,
    plotting: bool = False,
) -> np.ndarray:
    """Analyze trajectory deviation under sensing noise and control delay.

    Performs Monte Carlo simulation to analyze how sensing noise and control
    latency affect an agent's ability to follow reference trajectories.
    Returns the maximum deviations for each simulation run.

    Args:
        latency: Control update period in seconds (sensor-to-actuator delay)
        accuracy: Sensing accuracy (0-1) for state measurements.
            Can be scalar (applied to x,y only) or list of 5 values for
            [x, y, theta, v, omega]
        system: System type to analyze (default: "CAR")
        num_sims: Number of Monte Carlo simulation runs (default: 10)
        plotting: Whether to show trajectory plots (default: False)

    Returns:
        Array of maximum deviations (num_sims,) in meters

    Raises:
        NotImplementedError: If system type is not supported
        ValueError: If accuracy list has wrong dimensions

    Example:
        >>> devs = deviation(0.02, 0.9, num_sims=100)
        >>> print(f"Mean deviation: {np.mean(devs):.3f}m")
        >>> print(f"95th percentile: {np.percentile(devs, 95):.3f}m")
    """
    if system == "CAR":
        return _car_deviation(latency, accuracy, num_sims, plotting)
    else:
        raise NotImplementedError(f"Provided system '{system}' is not implemented.")


def _car_deviation(
    latency: float, accuracy: float | list[float], num_sims: int, plotting: bool
) -> np.ndarray:
    """Implementation of deviation analysis for car agents.

    Performs the core simulation and analysis for car agents following
    a figure-eight reference trajectory with sensing noise and control delays.

    Args:
        latency: Control update period in seconds
        accuracy: Sensing accuracy (scalar or 5-element list)
        num_sims: Number of simulation runs
        plotting: Whether to display trajectory plots

    Returns:
        Array of maximum trajectory deviations for each simulation

    Note:
        The reference trajectory consists of a figure-eight pattern with
        circular arcs and straight line segments, totaling 32 seconds duration.
    """
    control_period = latency

    if isinstance(accuracy, float):
        sensing_errors = [1 - accuracy] * 2 + [0.0, 0.0, 0.0]
    else:
        if len(accuracy) != 5:
            raise ValueError(
                f"Dimension of `accuracy` list must equal to {5}. Got {len(accuracy)}."
            )
        sensing_errors = [1 - a for a in accuracy]

    # Simulation configuration parameters
    SEED = 42
    time_step = 0.001

    # center and error for initial_set = [x, y, theta, v, omega]
    initial_set_c = (1.0, -0.5, 0, 0.0, 0.0)
    initial_set_e = (0.1,) * 2 + (0.1,) * 3

    traj = Trajectory(
        [
            ("circle", 8, (1, -0.5), (1, 0.5), (1, 0), "counterclockwise"),
            ("line", 8, (1, 0.5), (-1, 0.5)),
            ("circle", 8, (-1, 0.5), (-1, -0.5), (-1, 0), "counterclockwise"),
            ("line", 8, (-1, -0.5), (1, -0.5)),
        ]
    )
    time_horizon = traj.total_duration
    # End simulation configuration

    car1 = CarAgent(
        "car1",
        # Sensing errors only in x and y
        seed=SEED,
        control_period=control_period,
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

    traces = dubins_car.simulate_multi(
        time_horizon, time_step, max_height=6, num_sims=num_sims, seed=SEED
    )

    if plotting:
        fig = go.Figure()
        for t in traces:
            simulation_tree(t, fig=fig, print_dim_list=[0, 1, 2], map_type="fill")
        car1.plot_reference_trace(fig)
        fig.show()

    return trace_deviation(traces, car1)


if __name__ == "__main__":
    d = deviation(0.02, 0.5)
    print(np.max(d))
