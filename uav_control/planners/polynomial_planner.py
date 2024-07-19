from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import spatialmath as sm
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from spatialmath.base import (qconj, qdotb, qnorm, qvmul, rotx, roty, rotz,
                              skewa)

from uav_control.constants import R_B0_N, V_B0_N, e1_N
from uav_control.controllers.geometric_controller import (
    compute_unit_vector_ddot, compute_unit_vector_dot)
from uav_control.planners.polynomial_trajgen import PolynomialTrajectoryND


@dataclass
class QuadrotorPolynomialPlannerParams:
    waypoint_positions: np.ndarray  # waypoints to visit
    waypoint_times: np.ndarray  # times to reach each waypoint


class QuadrotorPolynomialPlanner(DiscreteTimeModel):
    def __init__(
        self, y0: Any, sample_rate: int, params: QuadrotorPolynomialPlannerParams
    ):
        """
        Initializes the QuadrotorPolynomialPlanner which is responsible for generating a minimum-snap polynomial trajectory for a quadrotor.
        This planner uses the provided initial state, sample rate, and parameters to create a trajectory based on waypoints and timepoints.

        Based on the paper:
            Minimum snap trajectory generation and control for quadrotors
            https://ieeexplore.ieee.org/document/5980409

        Args:
            y0 (Any): The initial state of the quadrotor system.
            sample_rate (int): The frequency (in Hz) at which the planner updates.
            params (QuadrotorPolynomialPlannerParams): Configuration parameters including waypoint positions and times.
        """
        super().__init__(
            y0, sample_rate, "dfb_planner", params, logging_level=LogLevel.INFO
        )

        self.polynomial_traj = PolynomialTrajectoryND(
            waypoints=params.waypoint_positions,
            timepoints=params.waypoint_times,
            order=7,
            n_constrained_end_derivs=4,
            minimize_order=2,
        )

        self.b1d_prev, self.b1d_dot_prev, self.b1d_ddot_prev = (
            e1_N,
            np.zeros(3),
            np.zeros(3),
        )

    def search_nearest_point(self, state: np.ndarray, t0):
        def grad(t):
            r_t, v_t = self.polynomial_traj(t, n_derivatives=1)
            return 2.0 * np.dot(r_t - state[R_B0_N], v_t)

        def objective(t):
            r_t = self.polynomial_traj(t, n_derivatives=0)
            return np.linalg.norm(r_t - state[R_B0_N]) ** 2

        t = t0

        for _ in range(100):
            gradient_eval, objective_eval = grad(t), objective(t)

            if np.allclose(gradient_eval, 0.0, atol=1e-6):
                gradient_eval = 2 * (np.random.rand() - 0.5) * 1e-6

            t_new = t - objective_eval / gradient_eval
            t_new = np.clip(
                t_new,
                self.polynomial_traj.timepoints[0],
                self.polynomial_traj.timepoints[-1],
            )

            if np.abs(t_new - t) < 1e-6:
                break

            t = t_new

        return t

        # t = t0 # Initial guess
        # grad_accum = 0.0 # Accumulated gradient
        # learning_rate = 0.5

        # grad_magnitude_eps = 1e-6 # Stopping criterion, gradient magnitude
        # max_iters = 200

        # def grad(t):
        #     r_t, v_t = self.polynomial_traj(t, n_derivatives=1)
        #     return 2.0 * np.dot(r_t - state[R_B0_N], v_t)

        # def objective(t):
        #     r_t = self.polynomial_traj(t, n_derivatives=0)
        #     return np.linalg.norm(r_t - state[R_B0_N]) ** 2

        # for i in range(max_iters):
        #     dJdt = grad(t)
        #     grad_accum += dJdt**2

        #     adjusted_lr = learning_rate / (np.sqrt(grad_accum) + 1e-6)

        #     if np.linalg.norm(dJdt) < 1e-6:
        #         t += learning_rate
        #     else:
        #         t += -adjusted_lr * dJdt

        #     if np.linalg.norm(dJdt) < grad_magnitude_eps:
        #         self.logger.info(f"Converged in {i} iterations")
        #         break
        #     else:
        #         t = t_new

        # self.logger.info(f"Final t: {t}, objective: {objective(t)}")
        # return t

    def discrete_dynamics(self, t: float, _y: Any) -> Any:
        """
        Returns the translational setpoints and desired 1st-body-axis direction at the current time.

        Args:
            t (float): The current time.
            _y (Any): Not applicable.

        Returns:
            List[np.ndarray], List[np.ndarray]:
                - The desired position, velocity, acceleration, jerk, and snap.
                - The desired 1st-body-axis direction and its 1st, 2nd derivatives.
        """
        state = self.input_models["quadrotor_state"].y
        t_nearest = t

        [r_b0_N_ref, v_b0_N_ref, a_b0_N_ref, j_b0_N_ref, s_b0_N_ref] = (
            self.polynomial_traj(t_nearest, n_derivatives=4)
        )

        v_b0_N_ref_planar = v_b0_N_ref[0:2]
        v_b0_N_ref_planar_norm = np.linalg.norm(v_b0_N_ref_planar)

        if v_b0_N_ref_planar_norm > 0.01:
            a_b0_N_ref_planar = a_b0_N_ref[0:2]
            j_b0_N_ref_planar = j_b0_N_ref[0:2]

            b1d = v_b0_N_ref_planar / v_b0_N_ref_planar_norm
            b1d_dot = compute_unit_vector_dot(v_b0_N_ref_planar, a_b0_N_ref_planar)
            b1d_ddot = compute_unit_vector_ddot(
                v_b0_N_ref_planar, a_b0_N_ref_planar, j_b0_N_ref_planar
            )

            self.b1d_prev, self.b1d_dot_prev, self.b1d_ddot_prev = (
                b1d,
                b1d_dot,
                b1d_ddot,
            )
        else:
            b1d, b1d_dot, b1d_ddot = (
                self.b1d_prev,
                self.b1d_dot_prev,
                self.b1d_ddot_prev,
            )

        return [r_b0_N_ref, v_b0_N_ref, a_b0_N_ref, j_b0_N_ref, s_b0_N_ref], [
            b1d,
            b1d_dot,
            b1d_ddot,
        ]
