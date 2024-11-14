from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import spatialmath as sm
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from spatialmath.base import qconj, qdotb, qnorm, qvmul, rotx, roty, rotz, skewa

from uav_control.constants import R_B0_N, V_B0_N, decompose_state, e1
from uav_control.planners.polynomial_trajgen import PolynomialTrajectoryND
from uav_control.utils.math import compute_unit_vector_ddot, compute_unit_vector_dot


class TrackingType(Enum):
    SPATIAL_TRACKING = 0
    TEMPORAL_TRACKING = 1


@dataclass
class QuadrotorPolynomialPlannerParams:
    waypoint_positions: np.ndarray  # waypoints to visit
    waypoint_times: np.ndarray  # times to reach each waypoint
    tracking_type: TrackingType = TrackingType.TEMPORAL_TRACKING


class QuadrotorPolynomialPlanner(DiscreteTimeModel):
    def __init__(self, y0: Any, sample_rate: int, params: QuadrotorPolynomialPlannerParams):
        """
        Initializes the QuadrotorPolynomialPlanner which is responsible for generating a minimum-snap polynomial trajectory for a quadrotor.
        This planner uses the provided initial state, sample rate, and parameters to create a trajectory based on waypoints and timepoints.

        Based on the paper:
            Minimum snap trajectory generation and control for quadrotors
            https://ieeexplore.ieee.org/document/5980409

        Parameters
        ----------
        y0 : Any
            The initial state of the quadrotor system.
        sample_rate : int
            The frequency (in Hz) at which the planner updates.
        params : QuadrotorPolynomialPlannerParams
            Configuration parameters including waypoint positions and times. Temporal tracking is the default tracking type,
            which means the planner will follow the trajectory based on the current time. Spatial tracking will follow the trajectory
            based on the closest waypoint to the current position.
        """
        super().__init__(y0, sample_rate, "polynomial_planner", params, logging_level=LogLevel.INFO)

        self.polynomial_traj = PolynomialTrajectoryND(
            waypoints=params.waypoint_positions,
            timepoints=params.waypoint_times,
            order=7,
            n_constrained_end_derivs=4,
            minimize_order=2,
        )

        if self.params.tracking_type == TrackingType.SPATIAL_TRACKING:
            self.closest_t_prev = params.waypoint_times[0]

        self.b1d_prev, self.b1d_dot_prev, self.b1d_ddot_prev = (
            e1,
            np.zeros(3),
            np.zeros(3),
        )

        # Log an info message with the initialized parameters and tracker type
        info_message = (
            f"QuadrotorPolynomialPlanner initialized with {params.tracking_type} tracking:\n"
        )
        for i, (waypoint, timepoint) in enumerate(
            zip(params.waypoint_positions, params.waypoint_times)
        ):
            info_message += f"\tWaypoint {i+1}: {waypoint} @ t={timepoint:.1}s\n"

        self.logger.info(info_message)

    def discrete_dynamics(self, t: float, _y: Any) -> Any:
        """
        Returns the translational setpoints and desired 1st-body-axis direction at the current time.

        Parameters
        ----------
        t : float
            Current simulation time
        _y : Any
            N/A

        Returns
        -------
        List[np.ndarray], List[np.ndarray]:
            The desired position, velocity, acceleration, jerk, and snap.
            The desired 1st-body-axis direction and its 1st, 2nd derivatives.
        """

        dynamics = self.input_models["quadrotor_state"]
        r_b0_N, q_NB, _v_b0_N, _omega_b0_B = decompose_state(dynamics.y)

        if self.params.tracking_type == TrackingType.SPATIAL_TRACKING:
            try:
                t_eval = self.polynomial_traj.closest_waypoint_t(
                    r_d=r_b0_N, bounds=(self.closest_t_prev, self.params.waypoint_times[-1])
                )
                self.closest_t_prev = t_eval
            except RuntimeError:
                self.logger.warning("No closest point found. Using previous closest point.")
                t_eval = self.closest_t_prev
        elif self.params.tracking_type == TrackingType.TEMPORAL_TRACKING:
            t_eval = t

        [r_b0_N_ref, v_b0_N_ref, a_b0_N_ref, j_b0_N_ref, s_b0_N_ref] = self.polynomial_traj(
            t=t_eval, n_derivatives=4
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
