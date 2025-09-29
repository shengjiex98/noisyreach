"""noisyreach: Noisy reachability analysis for autonomous agents.

This package provides tools for analyzing the reachability of autonomous agents
under sensing noise and control delays. The primary focus is on car-like agents
following trajectories with uncertain sensing capabilities.

The package includes:
- CarAgent: Autonomous car agents with tracking controllers
- CarMode: Operating modes for car agents
- Trajectory: Trajectory specification and following
- deviation: Core reachability analysis functions

Example:
    Basic usage for analyzing deviation from reference trajectory:

    >>> from noisyreach import deviation
    >>> max_deviation = deviation(latency=0.02, accuracy=0.9, system="CAR")
    >>> print(f"Maximum deviation: {max_deviation}")
"""

from noisyreach.car_agent import CarAgent, CarMode
from noisyreach.deviation import deviation
from noisyreach.trajectory import Trajectory

__all__ = ["CarAgent", "CarMode", "deviation", "Trajectory"]
