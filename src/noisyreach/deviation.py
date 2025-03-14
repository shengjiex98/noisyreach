import numpy as np
import plotly.graph_objects as go
from verse import Scenario, ScenarioConfig
from verse.analysis import AnalysisTree
from verse.plotter.plotter2D import simulation_tree

from noisyreach import CarAgent, CarMode, Trajectory

SEED = 42


def trace_deviation(traces: list[AnalysisTree], agent: CarAgent):
    traces = np.asarray([trace.root.trace[agent.id] for trace in traces])
    reference_trace = agent.reference_trace(list(traces[0, :, 0]))

    diff = traces[:, :, 1:3] - np.expand_dims(reference_trace, 0)[:, :, 1:3]

    return np.max(
        np.apply_along_axis(lambda row: np.hypot(row[0], row[1]), -1, diff),
        axis=1,
        keepdims=False,
    )


def deviation(latency: float, accuracy: float, system="Car", n_simulations=10):
    control_period = latency
    sensing_errors = [1 - accuracy] * 2 + [0.0, 0.0, 0.0]

    # >>>>>>>>> Simulation Parameters >>>>>>>>>
    time_step = 0.001

    # center and error for initial_set = [x, y, theta, v, omega]
    initial_set_c = (2.0, -1.0, 0, 0.0, 0.0)
    initial_set_e = (0.2,) * 2 + (0.2,) * 3

    traj = Trajectory(
        [
            ("circle", 15, (2, -1), (2, 1), (2, 0), "counterclockwise"),
            ("line", 15, (2, 1), (-2, 1)),
            ("circle", 15, (-2, 1), (-2, -1), (-2, 0), "counterclockwise"),
            ("line", 15, (-2, -1), (2, -1)),
        ]
    )
    time_horizon = traj.total_duration
    # <<<<<<<<< Simulation Parameters <<<<<<<<<

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
        time_horizon, time_step, max_height=6, n_sims=n_simulations, seed=SEED
    )

    fig = go.Figure()
    for t in traces:
        simulation_tree(t, fig=fig, print_dim_list=[0, 1, 2], map_type="fill")
    car1.plot_reference_trace(fig)
    fig.show()

    return trace_deviation(traces, car1)


if __name__ == "__main__":
    d = deviation(0.02, 0.5)
    print(np.max(d))
