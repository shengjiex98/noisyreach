"""Linear system simulation with LQR control and sensing noise.

This module provides a LinearSim class for simulating linear time-invariant systems
with discrete-time LQR control and optional sensing noise. The system supports both
automatic LQR controller design and manual controller specification.
"""

from decimal import Decimal

import control as ct
import numpy as np


class LinearSim:
    """Linear system simulator with discrete-time control and sensing noise.

    This class simulates linear time-invariant systems of the form:
        dx/dt = Ax + Bu (continuous dynamics)
        y = Cx + Du (output equation)

    The system uses discrete-time control with configurable sampling periods
    and supports sensing noise through multiplicative Gaussian errors.

    Attributes:
        METHODS: List of supported controller design methods ['lqr', 'manual']
    """

    METHODS = ["lqr", "manual"]

    def __init__(
        self,
        id,
        A: np.ndarray,
        B: np.ndarray,
        C=None,
        D=None,
        method="lqr",
        period=None,
        sensing_error_std: np.ndarray | None = None,
        seed=None,
        **kwargs,
    ):
        """Initialize the linear system simulator.

        Args:
            id: Unique identifier for this simulator instance
            A: State transition matrix (nx x nx)
            B: Input matrix (nx x nu)
            C: Output matrix (ny x nx). Defaults to identity if None
            D: Feedthrough matrix (ny x nu). Defaults to zeros if None
            method: Controller design method ('lqr' or 'manual')
            period: Control sampling period in seconds
            sensing_error_std: Standard deviations for sensing noise (length nx)
            seed: Random seed for reproducible noise generation
            **kwargs: Additional arguments for controller design:
                - For 'lqr': Q (state cost matrix), R (input cost matrix)
                - For 'manual': K (controller gain matrix)

        Raises:
            ValueError: If sensing_error_std length doesn't match state dimension
        """
        if C or D:
            print("Caution: custom C and D are not well tested yet.")
        nx = A.shape[0]
        C = C if C is not None else np.eye(nx)
        D = (
            D if D is not None else np.zeros((C.shape[0], B.shape[1]))
        )  # shape: (ny, nu)

        # if K and K.shape != (nu, nx):
        #     raise ValueError(f"K must have shape (nu={nu}, nx={nx}). Got {K.shape}.")
        # if method not in LinearSim.METHODS:
        #     raise ValueError(f"{method} not in supported methods ({LinearSim.METHODS}).")

        if sensing_error_std is not None and len(sensing_error_std) != nx:
            raise ValueError(
                f"sensing_error_std must have length nx={nx}. Got {len(sensing_error_std)}."
            )

        self.id = id
        self.csys = ct.StateSpace(A, B, C, D)

        self.sensing_error_std = sensing_error_std

        self.seed = seed
        self.nonce = 0

        self.period = period
        self.update_controller(method, **kwargs)

    def _sensor(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply sensing noise to state measurements.

        Applies multiplicative Gaussian noise: sensed_state = (1 + noise) * true_state
        where noise ~ N(0, sensing_error_std).

        Args:
            state: True system state (nx,)
            rng: Random number generator for noise sampling

        Returns:
            Noisy state measurement (nx,)
        """
        if self.sensing_error_std is None:
            return state
        else:
            error = rng.normal(0, self.sensing_error_std)
            return (1 + error) * state

    def _dynamics(self, _t: float, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute continuous-time system dynamics.

        Implements dx/dt = Ax + Bu for the linear system.

        Args:
            _t: Time (unused for time-invariant systems)
            state: Current state vector (nx,)
            u: Control input vector (nu,)

        Returns:
            State derivative dx/dt (nx,)
        """
        return self.csys.A @ state + self.csys.B @ u

    def update_period(self, period: float) -> None:
        """Update the control sampling period.

        Args:
            period: New control sampling period in seconds

        Warning:
            For manual controller design, the gain matrix K is not automatically
            updated for the new period and may need manual adjustment.
        """
        if self.method == "manual":
            print(
                "Warning: controller design method is set to 'manual'; "
                + "controller gain K not automatically updated for the new period."
            )

        self.period = period

    def update_controller(self, method: str, **kwargs) -> None:
        """Update or redesign the controller.

        Args:
            method: Controller design method ('lqr' or 'manual')
            **kwargs: Method-specific parameters:
                - For 'lqr': Q (state cost matrix), R (input cost matrix)
                - For 'manual': K (controller gain matrix)

        Raises:
            ValueError: If method is not supported or required parameters missing
        """
        if method == "lqr":
            Q = kwargs.get("Q", np.eye(self.csys.nstates))
            R = kwargs.get("R", np.eye(self.csys.ninputs))
            K, _, _ = ct.dlqr(ct.c2d(self.csys, self.period), Q, R)
        elif method == "manual":
            K = kwargs.get("K", None)
            if K is None:
                raise ValueError("K must be provided for manual controller update.")
            elif K.shape != (self.csys.ninputs, self.csys.nstates):
                raise ValueError(
                    f"K should have shape {(self.csys.ninputs, self.csys.nstates)}. Got {K.shape}."
                )
        else:
            raise ValueError(
                f"{method} not in supported methods ({LinearSim.METHODS})."
            )

        self.method = method
        self.K = K

    def simulate(
        self, initial_set: np.ndarray, time_horizon: float, time_step: float
    ) -> np.ndarray:
        """Simulate the closed-loop system with discrete-time control.

        Simulates the system with one control period sensor-to-actuator delay,
        meaning control inputs are computed based on measurements from the previous
        control period.

        Args:
            initial_set: Initial state vector (nx,)
            time_horizon: Total simulation time in seconds
            time_step: Simulation time step in seconds

        Returns:
            Simulation trace array (n_points, ny+1) where:
            - Column 0: time stamps
            - Columns 1:ny+1: system outputs y = Cx + Du

        Raises:
            ValueError: If initial_set dimension doesn't match system state dimension

        Warning:
            Issues warning if control period is not a multiple of simulation time step
        """
        if len(initial_set) != self.csys.nstates:
            raise ValueError(f"initial_set must have length nx={self.csys.nstates}.")

        # TODO: currently only using Decimal for warning, not computation. Consider
        # using Decimals also for calculation, or other ways to help with precision or
        # rounding errors.
        if Decimal(str(self.period)) % Decimal(str(time_step)) != 0:
            print(
                "Warning: period must be multiples of simulation's time_step."
                + f"Got {self.period, time_step}"
            )

        # All timestamps to simulate
        t_eval = np.arange(0, time_horizon + time_step, time_step)
        n_points = t_eval.shape[0]

        # All theoretical control instants.
        control_instants = np.append(
            np.arange(0, t_eval[-1], self.period), [t_eval[-1]]
        )

        # Find indices of timestamps where the practical calculations of control
        # inputs should occur.
        control_indices = np.searchsorted(t_eval, control_instants, "left")

        trace = np.zeros((n_points, self.csys.noutputs + 1))
        trace[:, 0] = t_eval

        # Set seed for deterministic sampling in `self._sensor()`. Incrementing
        # `self.nonce` each time ensures different sampling when running `simulate()`
        # multiple times.
        if self.seed:
            rng = np.random.default_rng(self.seed + self.nonce)
            self.nonce += 1
        else:
            rng = np.random.default_rng()

        # Linear calculations assuming one period sensor-to-actuator delay
        # Control input u[k] is based on measurement from previous control period
        dsys = ct.c2d(self.csys, time_step)
        s = initial_set
        u = np.zeros(self.csys.ninputs)
        for i_start, i_end in zip(control_indices[:-1], control_indices[1:]):
            for i in range(i_start, i_end):
                trace[i, 1:] = dsys.C @ s + dsys.D @ u
                s = dsys.A @ s + dsys.B @ u
            u = -self.K @ self._sensor(s, rng)
        trace[-1, 1:] = dsys.C @ s + dsys.D @ u

        return trace
