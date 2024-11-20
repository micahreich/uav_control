from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import spatialmath as sm
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from spatialmath.base import (qconj, qdotb, qnorm, qvmul, rotx, roty, rotz,
                              skewa)

from uav_control.constants import R_B0_N, V_B0_N, decompose_state, e1, e2


@dataclass
class QuadrotorWaypointPlannerParams:
    waypoint_positions: np.ndarray  # waypoints to visit
    waypoint_times: Optional[np.ndarray]  # times to reach each waypoint


class QuadrotorWaypointPlanner(DiscreteTimeModel):
    def __init__(
        self,
        y0: Any,
        sample_rate: int,
        params: QuadrotorWaypointPlannerParams,
        name: str = "waypoint_planner",
        logging_level: LogLevel = LogLevel.ERROR,
    ):
        """
        Initializes a QuadrotorWaypointPlanner object. Given some waypoints and times, the planner will
        switch between tracking the next desired waypoint based on the current time.

        Args:
            y0 (Any): The initial state of the quadrotor.
            sample_rate (int): The frequency (in Hz) at which the planner updates.
            params (QuadrotorWaypointPlannerParams): Configuration parameters including waypoint positions and times.
        """
        super().__init__(y0, sample_rate, name, params, logging_level=logging_level)
        self.b1d_prev = e1
        self.waypoint_idx = 0

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

        dynamics = self.input_models["quadrotor_state"]
        r_b0_N, _, _, _ = decompose_state(dynamics.y)

        close_to_waypoint_thresh_m = 0.5
        v_pos_to_waypoint = self.params.waypoint_positions[idx] - r_b0_N

        if self.params.waypoint_times is not None:
            t_verified = np.clip(t, self.params.waypoint_times[0], self.params.waypoint_times[-1])
            idx = np.searchsorted(self.params.waypoint_times, t_verified, side="left")
        else:
            if np.linalg.norm(v_pos_to_waypoint) < close_to_waypoint_thresh_m:
                self.waypoint_idx = max(
                    self.waypoint_idx + 1, len(self.params.waypoint_positions) - 1
                )
                idx = self.waypoint_idx

                v_pos_to_waypoint = self.params.waypoint_positions[idx] - r_b0_N

        planar_projection = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

        # If the quadrotor is close to the waypoint, keep the previous yaw direction
        if np.linalg.norm(v_pos_to_waypoint) < 0.5:
            b1d = self.b1d_prev
        else:
            b1d = planar_projection @ (v_pos_to_waypoint / np.linalg.norm(v_pos_to_waypoint))

        self.b1d_prev = b1d

        return [
            self.params.waypoint_positions[idx],
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        ], [b1d, np.zeros(3), np.zeros(3)]
