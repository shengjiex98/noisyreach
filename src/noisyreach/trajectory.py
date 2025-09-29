"""Trajectory specification and following for autonomous agents.

This module provides a Trajectory class that supports efficient generation of
state sequences for agents following piecewise trajectories. Trajectories can
consist of line segments and circular arcs, allowing for complex path following
behaviors.

The trajectory system is designed for real-time use with strictly increasing
time queries, using generator-based state computation to minimize memory usage.

Example:
    Create a trajectory with line and circle segments:

    >>> traj = Trajectory(
    ...     [
    ...         ("line", 5.0, (0, 0), (10, 0)),  # Straight line for 5 seconds
    ...         ("circle", 3.14, (10, 0), (10, 2), (10, 1), "counterclockwise"),
    ...     ]
    ... )
    >>> posture, velocity = traj.get_state(2.5)  # Get state at t=2.5s
"""

import math


class Trajectory:
    """Efficient trajectory following with piecewise segments.

    A trajectory consists of ordered segments, each specifying a motion primitive
    (line or circle) with duration and geometric parameters. The trajectory can
    be queried for position and velocity at any time.

    The implementation uses a generator-based approach for memory efficiency,
    maintaining minimal state while supporting real-time trajectory following.
    """

    def __init__(self, traj_specs: list, tol: float = 1e-7):
        """Initialize trajectory with segment specifications.

        Args:
            traj_specs: List of trajectory segment specifications. Each spec is either:
                - ('line', duration, (x0, y0), (x1, y1)): Linear motion
                - ('circle', duration, (x0, y0), (x1, y1), (x_center, y_center), direction):
                  Circular arc motion where direction is 'clockwise' or 'counterclockwise'
            tol: Tolerance for segment boundary checks (default: 1e-7)

        Example:
            >>> specs = [
            ...     ("line", 2.0, (0, 0), (2, 0)),  # 2m line in 2s
            ...     (
            ...         "circle",
            ...         3.14,
            ...         (2, 0),
            ...         (2, 2),
            ...         (2, 1),
            ...         "counterclockwise",
            ...     ),  # Half circle
            ... ]
            >>> traj = Trajectory(specs)
        """
        self.traj_specs = traj_specs
        self.tol = tol
        # No cumulative times or optimized segments are stored.
        self.reset()

    @property
    def total_duration(self) -> float:
        return sum([spec[1] for spec in self.traj_specs])

    def reset(self) -> None:
        """Reset trajectory state for new time sequence.

        Resets the internal generator state to allow querying from t=0 again.
        Call this when you need to restart trajectory following or query
        with non-increasing time values.
        """
        self.last_t = -math.inf
        self._gen = self._state_generator()
        next(self._gen)  # Prime the generator

    def get_state(
        self, t: float | None
    ) -> tuple[tuple[float, float, float], tuple[float, float]] | None:
        """Get trajectory state at specified time.

        Retrieves the position, orientation, and velocity at time t. Time values
        must be strictly increasing for efficiency, or reset() should be called.

        Args:
            t: Time in seconds (must be >= previous query time)

        Returns:
            Tuple containing:
            - posture: (x, y, theta) - position and heading angle
            - velocity: (v, omega) - linear and angular velocities
            Returns None if t is None or exceeds trajectory duration.

        Raises:
            ValueError: If t exceeds the total trajectory duration
            Warning: If t < previous t (inefficient, triggers reset)

        Example:
            >>> traj = Trajectory([("line", 2.0, (0, 0), (2, 0))])
            >>> pos, vel = traj.get_state(1.0)
            >>> print(f"Position: {pos}, Velocity: {vel}")
        """
        if t is None:
            return None
        if t < self.last_t:
            self.reset()
            raise Warning(
                "Passing a t value that is smaller than the previous t value is inefficient. Consider calling `get_state()` with non-decreasing t values. Use `reset()` to start from t=0."
            )
        return self._gen.send(t)

    def _state_generator(self):
        """Internal generator for efficient state computation.

        Generator that processes trajectory segments on-demand, maintaining minimal
        state while supporting real-time queries. Expects strictly increasing time
        values for optimal performance.

        Yields:
            Tuple of ((x, y, theta), (v, omega)) for each received time value

        Raises:
            ValueError: When time exceeds trajectory duration
        """
        current_seg_start = 0.0

        # Prime the generator: wait for the first time value.
        t = yield

        for spec in self.traj_specs:
            # Skip unused segments
            seg_end = current_seg_start + spec[1]
            if t > seg_end + self.tol:
                current_seg_start = seg_end
                continue

            posture_f, velocity_f = Trajectory._get_state_functions(spec)

            while True:
                # If t has advanced beyond this segment, break to process next segment.
                if t > seg_end + self.tol:
                    current_seg_start = seg_end
                    break

                t_relative = t - current_seg_start

                # Yield the state and wait for the next t
                t = yield posture_f(t_relative), velocity_f(t_relative)

        # If we run out of segments, any further t values are out of bounds.
        while True:
            t = yield None
            raise ValueError("Time t exceeds the trajectory duration.")

    @staticmethod
    def _get_state_functions(spec: tuple):
        if spec[0] == "line":
            return Trajectory._line_f(spec)
        elif spec[0] == "circle":
            return Trajectory._circle_f(spec)
        else:
            raise NotImplementedError("Segment spec passed is not implemented.")

    @staticmethod
    def _line_f(spec: tuple):
        """Generate state functions for linear trajectory segments.

        Args:
            spec: Line specification tuple ('line', duration, (x0, y0), (x1, y1))

        Returns:
            Tuple of (posture_function, velocity_function)
        """
        duration = spec[1]
        (x0, y0) = spec[2]
        (x1, y1) = spec[3]
        vx = (x1 - x0) / duration
        vy = (y1 - y0) / duration
        theta = math.atan2(vy, vx)
        v = math.hypot(vx, vy)
        omega = 0.0

        def posture_f(t):
            return (x0 + vx * t, y0 + vy * t, theta)

        def velocity_f(t):
            return (v, omega)

        return posture_f, velocity_f

    @staticmethod
    def _circle_f(spec: tuple):
        """Generate state functions for circular trajectory segments.

        Args:
            spec: Circle specification tuple:
                ('circle', duration, (x0, y0), (x1, y1), (x_center, y_center), direction)

        Returns:
            Tuple of (posture_function, velocity_function)

        Raises:
            ValueError: If direction is not 'clockwise' or 'counterclockwise'
        """
        direction = spec[5]
        if direction not in ["clockwise", "counterclockwise"]:
            raise ValueError(
                "Invalid circle direction: must be 'clockwise' or 'counterclockwise'"
            )
        duration = spec[1]
        (x0, y0) = spec[2]
        (x1, y1) = spec[3]
        (xc, yc) = spec[4]
        phi_start = math.atan2(y0 - yc, x0 - xc)
        phi_end = math.atan2(y1 - yc, x1 - xc)
        dphi = phi_end - phi_start
        if direction == "counterclockwise" and dphi < 0:
            dphi += 2 * math.pi
        elif direction == "clockwise" and dphi > 0:
            dphi -= 2 * math.pi
        R = math.hypot(x0 - xc, y0 - yc)
        omega = dphi / duration
        v = omega * R

        def posture_f(t):
            phi = phi_start + (t / duration) * dphi
            return (
                xc + R * math.cos(phi),
                yc + R * math.sin(phi),
                phi
                + (math.pi / 2 if direction == "counterclockwise" else -math.pi / 2),
            )

        def velocity_f(t):
            return (v, omega)

        return posture_f, velocity_f


# --- Example Usage ---
if __name__ == "__main__":
    # Define a trajectory with two segments:
    # 1. A straight line from (0,0) to (10,0) over 5 seconds.
    # 2. A quarter-circle arc from (10,0) to (15,5) over the next 5 seconds,
    #    with center (10,5) and counterclockwise motion.
    traj_specs = [
        ("line", 5.0, (0, 0), (10, 0)),
        ("circle", 5.0, (10, 0), (15, 5), (10, 5), "counterclockwise"),
    ]
    traj = Trajectory(traj_specs)

    # Call get_state with strictly increasing t values.
    times = [0, 2.5, 5.0, 7.5, 10.0]
    for t in times:
        posture, velocity = traj.get_state(t)
        print(f"t = {t:.2f} s: posture = {posture}, velocity = {velocity}")

    # Reset to allow lower t values.
    traj.reset()
    print("\nAfter reset:")
    for t in [0, 1.0, 2.0]:
        posture, velocity = traj.get_state(t)
        print(f"t = {t:.2f} s: posture = {posture}, velocity = {velocity}")
