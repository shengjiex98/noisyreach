import math


class Trajectory:
    def __init__(self, traj_specs, tol=1e-7):
        """
        traj_specs: list of trajectory specifications. Each spec is either:
          ('line', duration, (x0, y0), (x1, y1))
          or
          ('circle', duration, (x0, y0), (x1, y1), (x_center, y_center), dir)
        tol: tolerance used when checking segment boundaries.
        """
        self.traj_specs = traj_specs
        self.tol = tol
        # No cumulative times or optimized segments are stored.
        self.reset()

    @property
    def total_duration(self) -> float:
        return sum([spec[1] for spec in self.traj_specs])

    def reset(self) -> None:
        """
        Resets the internal state so that lower t values can be used again.
        """
        self.last_t = -math.inf
        self._gen = self._state_generator()
        next(self._gen)  # Prime the generator

    def get_state(self, t) -> tuple[tuple[float], tuple[float]]:
        """
        Sends a strictly increasing time t to the state generator and returns the state:
            ((x, y, theta), (v, omega))
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
        """
        A generator that expects strictly increasing time values.
        It iterates over the trajectory specs on the fly without storing extra arrays.
        Maintains only the current segment and its start time.
        Yields a tuple ((x, y, theta), (v, omega)) for each received time t.
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
    def _get_state_functions(spec):
        if spec[0] == "line":
            return Trajectory._line_f(spec)
        elif spec[0] == "circle":
            return Trajectory._circle_f(spec)
        else:
            raise NotImplementedError("Segment spec passed is not implemented.")

    @staticmethod
    def _line_f(spec):
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
    def _circle_f(spec):
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
