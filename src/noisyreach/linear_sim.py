from decimal import Decimal

import control as ct
import numpy as np


class LinearSim:
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
        if C or D:
            print("Caution: custom C and D are not well tested yet.")
        C = C if C else np.eye(nx := A.shape[0])
        D = D if D else np.zeros((C.shape[0], B.shape[1]))  # shape: (ny, nu)

        # if K and K.shape != (nu, nx):
        #     raise ValueError(f"K must have shape (nu={nu}, nx={nx}). Got {K.shape}.")
        # if method not in LinearSim.METHODS:
        #     raise ValueError(f"{method} not in supported methods ({LinearSim.METHODS}).")

        if sensing_error_std and len(sensing_error_std) != nx:
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

    def _sensor(self, state: np.ndarray, rng: np.random.BitGenerator):
        if self.sensing_error_std is None:
            return state
        else:
            error = rng.normal(0, self.sensing_error_std)
            return (1 + error) * state

    # Continuous dynamics
    def _dynamics(self, t: float, state: np.ndarray, u: np.ndarray):
        return self.csys.A @ state + self.csys.B @ u

    def update_period(self, period: float):
        if self.method == "manual":
            print(
                "Warning: controller design method is set to 'manual'; "
                + "controller gain K not automatically updated for the new period."
            )

        self.period = period

    def update_controller(self, method: str, **kwargs) -> None:
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

    def simulate(self, initial_set: np.ndarray, time_horizon: float, time_step: float):
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

        # Find indecies of timestamps where the practical calculations of control
        # inputs should occur.
        control_indices = np.searchsorted(t_eval, control_instants, "left")

        trace = np.zeros((n_points, self.csys.noutputs + 1))
        trace[:, 0] = t_eval

        # Set seed for deterministic sampling in `self.sensor()`. Incrementing
        # `self.nonce` each time ensures different sampling when running `simulate()`
        # multiple times.
        if self.seed:
            rng = np.random.default_rng(self.seed + self.nonce)
            self.nonce += 1
        else:
            rng = np.random.default_rng()

        # Linear calculations assuming one period sensor-to-actuator delay
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
