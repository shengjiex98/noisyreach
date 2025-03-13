import plotly.graph_objects as go
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import simulation_tree

from noisyreach import CarAgent, CarMode, Trajectory

if __name__ == "__main__":
    # >>>>>>>>> Simulation Parameters >>>>>>>>>
    time_step = 0.001
    control_period = 0.1

    # center and error for initial_set =[x, y, theta, v, omega]
    initial_set_c = (2.0, -1.0, 0, 0.0, 0.0)
    initial_set_e = (0.2,) * 2 + (0.2,) * 3

    accuracy = 0.8
    sensing_errors = [1 - accuracy] * 2 + [0.0, 0.0, 0.0]

    traj = Trajectory(
        [
            ("circle", 15, (2, -1), (2, 1), (2, 0), "counterclockwise"),
            ("line", 15, (2, 1), (-2, 1)),
            ("circle", 15, (-2, 1), (-2, -1), (-2, 0), "counterclockwise"),
            ("line", 15, (-2, -1), (2, -1)),
        ]
    )
    time_horizon = traj.total_duration
    n_simulations = 10
    # <<<<<<<<< Simulation Parameters <<<<<<<<<

    car1 = CarAgent(
        "car1",
        # Sensing errors only in x and y
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
        time_horizon, time_step, max_height=6, n_sims=n_simulations
    )

    fig = go.Figure()
    # simulation_tree(traces[0], None, fig, 1, 2, [0, 1, 2], "fill", "trace")
    for trace in traces:
        simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    car1.plot_reference_trace(fig, n_points=70)
    fig.update_layout(title=f"T={car1.control_period:.3f} Acc={accuracy:.2f}")
    fig.show()
