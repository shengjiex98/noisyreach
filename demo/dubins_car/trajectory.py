import math


def compute_line_state(spec, s):
    """
    For a line spec ('line', duration, (x0,y0), (x1,y1)) and a time s in [0, duration],
    compute the vehicle state:
      - Position: linear interpolation from (x0, y0) to (x1, y1)
      - Velocity: constant vector ( (x1-x0)/duration, (y1-y0)/duration )
      - Orientation: angle of the velocity vector
      - Angular velocity: zero (since the line is straight)
    Returns:
      posture = (x, y, theta)
      velocity = (v, omega)
    """
    duration = spec[1]
    (x0, y0) = spec[2]
    (x1, y1) = spec[3]
    fraction = s / duration
    x = x0 + fraction * (x1 - x0)
    y = y0 + fraction * (y1 - y0)

    dx = (x1 - x0) / duration
    dy = (y1 - y0) / duration
    v = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    omega = 0.0

    posture = (x, y, theta)
    velocity = (v, omega)
    return posture, velocity


def compute_circle_state(spec, s):
    """
    For a circle spec ('circle', duration, (x0,y0), (x1,y1), (x_center, y_center), dir)
    and a time s in [0, duration]:

      - Compute the polar angles of the start and end points relative to the center.
      - Adjust the angular difference dphi so that for a 'counterclockwise' arc dphi > 0
        (adding 2π if necessary), or for a 'clockwise' arc dphi < 0 (subtracting 2π if needed).
      - Parameterize the arc: φ(s) = φ_start + (s/duration)*dphi.
      - The radius R is the distance from the center to the start point.
      - Position is (x_center + R*cos(φ), y_center + R*sin(φ)).
      - Differentiating, the velocity vector is
              dx/dt = -R*sin(φ) * (dphi/duration),
              dy/dt =  R*cos(φ) * (dphi/duration).
      - The speed is |R*dphi/duration| and the orientation is obtained from atan2(dy/dt, dx/dt).
      - Angular velocity omega is dphi/duration.
    Returns:
      posture = (x, y, theta)
      velocity = (v, omega)
    """
    duration = spec[1]
    (x0, y0) = spec[2]
    (x1, y1) = spec[3]
    (x_center, y_center) = spec[4]
    direction = spec[5]

    # Compute start and end angles (phi) relative to the center.
    phi_start = math.atan2(y0 - y_center, x0 - x_center)
    phi_end = math.atan2(y1 - y_center, x1 - x_center)

    # Determine the angle change based on the desired motion direction.
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

    # Compute current angle along the arc.
    phi = phi_start + (s / duration) * dphi

    # Radius is assumed constant.
    R = math.hypot(x0 - x_center, y0 - y_center)

    # Position on the circle.
    x = x_center + R * math.cos(phi)
    y = y_center + R * math.sin(phi)
    theta = phi + (math.pi if direction == "counterclockwise" else -math.pi) / 2

    # Angular velocity is constant.
    omega = dphi / duration
    v = omega * R

    posture = (x, y, theta)
    velocity = (v, omega)
    return posture, velocity


def convert_trajectory_to_state(traj_specs, t, tol=1e-7):
    """
    Given a piecewise-smooth trajectory defined as a list of specs and a time t,
    compute the vehicle's posture and velocity.

    The trajectory is a list of specs, where each spec is one of:
       ('line', duration, (x0, y0), (x1, y1))
       ('circle', duration, (x0, y0), (x1, y1), (x_center, y_center), dir)

    The overall time of the trajectory is the sum of the durations.

    When t is exactly (within tol) at a segment boundary, the derivatives (and hence theta)
    are computed from the later segment.

    Returns:
       posture = (x, y, theta)
       velocity = (v, omega)
    """
    # Compute cumulative ending times.
    cumulative_times = []
    cumulative = 0.0
    for spec in traj_specs:
        cumulative += spec[1]
        cumulative_times.append(cumulative)

    if t < 0 or t > cumulative_times[-1]:
        raise ValueError("Time t is out of the trajectory bounds.")

    # Determine which segment t falls into.
    segment_index = None
    segment_start_time = 0.0
    for i, end_time in enumerate(cumulative_times):
        if t < end_time - tol:
            segment_index = i
            break
        elif abs(t - end_time) < tol:
            # At a segment transition, use the later segment if available.
            if i + 1 < len(traj_specs):
                segment_index = i + 1
                segment_start_time = end_time
            else:
                segment_index = i
            break
        segment_start_time = end_time

    # In case t is exactly the final time.
    if segment_index is None:
        segment_index = len(traj_specs) - 1
        segment_start_time = cumulative_times[-2] if len(cumulative_times) > 1 else 0.0

    # Relative time in the chosen segment.
    relative_t = t - segment_start_time
    spec = traj_specs[segment_index]

    if spec[0] == "line":
        return compute_line_state(spec, relative_t)
    elif spec[0] == "circle":
        return compute_circle_state(spec, relative_t)
    else:
        raise ValueError("Unknown segment type: " + str(spec[0]))


# --- Example Usage ---
if __name__ == "__main__":
    # Define a trajectory with two segments:
    # 1. A straight line from (0,0) to (10,0) over 5 seconds.
    # 2. A quarter-circle arc from (10,0) to (15,5) over the next 5 seconds.
    #    The circle has center (10,5), so that:
    #       - The start point (10,0) gives a polar angle of -pi/2.
    #       - The end point (15,5) gives a polar angle of 0.
    #    With "counterclockwise" motion, the vehicle sweeps an angle of pi/2.

    traj = [
        ("line", 5.0, (0, 0), (10, 0)),
        ("circle", 5.0, (10, 0), (15, 5), (10, 5), "counterclockwise"),
    ]

    # Sample times: 0, 2.5, 5.0, 7.5, and 10.0 seconds.
    sample_times = [0, 2.5, 5.0, 7.5, 10.0]
    for t in sample_times:
        posture, velocity = convert_trajectory_to_state(traj, t)
        print(f"t = {t:.2f} s: posture = {posture}, velocity = {velocity}")
