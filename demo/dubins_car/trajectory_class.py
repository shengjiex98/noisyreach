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

    def _state_generator(self):
        """
        A generator that expects strictly increasing time values.
        It iterates over the trajectory specs on the fly without storing extra arrays.
        Maintains only the current segment and its start time.
        Yields a tuple ((x, y, theta), (v, omega)) for each received time t.
        """
        last_t = -float("inf")
        current_segment = 0
        current_seg_start = 0.0  # Global time when the current segment starts

        # Prime the generator: wait for the first time value.
        t = yield

        # Process each segment in order.
        while current_segment < len(self.traj_specs):
            spec = self.traj_specs[current_segment]
            duration = spec[1]
            seg_end = current_seg_start + duration

            # Compute any per-segment constants on the fly.
            if spec[0] == "line":
                (x0, y0) = spec[2]
                (x1, y1) = spec[3]
                vx = (x1 - x0) / duration
                vy = (y1 - y0) / duration
                theta = math.atan2(vy, vx)
                v = math.hypot(vx, vy)
                omega = 0.0

                def posture(t):
                    return (x0 + vx * t, y0 + vy * t, theta)

                def velocity(t):
                    return (v, omega)
            elif spec[0] == "circle":
                (x0, y0) = spec[2]
                (x1, y1) = spec[3]
                (xc, yc) = spec[4]
                direction = spec[5]
                phi_start = math.atan2(y0 - yc, x0 - xc)
                phi_end = math.atan2(y1 - yc, x1 - xc)
                dphi = phi_end - phi_start
                if direction == "counterclockwise":
                    if dphi < 0:
                        dphi += 2 * math.pi
                elif direction == "clockwise":
                    if dphi > 0:
                        dphi -= 2 * math.pi
                else:
                    raise ValueError(
                        "Invalid circle direction: must be 'clockwise' or 'counterclockwise'"
                    )
                R = math.hypot(x0 - xc, y0 - yc)

                def posture(t):
                    phi = phi_start + (t / duration) * dphi
                    return (
                        xc + R * math.cos(phi),
                        yc + R * math.sin(phi),
                        phi
                        + (
                            math.pi / 2
                            if direction == "counterclockwise"
                            else -math.pi / 2
                        ),
                    )

                omega = dphi / duration
                v = omega * R

                def velocity(t):
                    return (v, omega)
            else:
                raise ValueError("Unknown spec type: " + spec[0])

            # Process times that fall into the current segment.
            while True:
                if t is None:
                    t = yield None  # Wait if no time is provided

                # Enforce strictly increasing time.
                if t <= last_t:
                    raise ValueError(
                        f"Time t must be strictly increasing. Received t = {t} but last t was {last_t}."
                    )

                # If t has advanced beyond this segment, break to process next segment.
                if (
                    t > seg_end + self.tol
                    or abs(t - seg_end) < self.tol
                    and current_segment < len(self.traj_specs) - 1
                ):
                    break

                relative_t = t - current_seg_start
                last_t = t

                # Yield the state and wait for the next t
                t = yield posture(relative_t), velocity(relative_t)

            # t is now past the current segment; move on.
            current_segment += 1
            current_seg_start = seg_end

        # If we run out of segments, any further t values are out of bounds.
        while True:
            t = yield None
            raise ValueError("Time t exceeds the trajectory duration.")

    def get_state(self, t):
        """
        Sends a strictly increasing time t to the state generator and returns the state:
            ((x, y, theta), (v, omega))
        """
        return self._gen.send(t)

    def reset(self):
        """
        Resets the internal state so that lower t values can be used again.
        """
        self._gen = self._state_generator()
        next(self._gen)  # Prime the generator


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
