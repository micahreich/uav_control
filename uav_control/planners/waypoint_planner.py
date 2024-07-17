import numpy as np
import spatialmath as sm
from spatialmath.base import qnorm, qconj, qdotb, qvmul, rotx, roty, rotz, skewa

from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel

from uav_control.constants import R_B0_N, V_B0_N, e1_N, e2_N, decompose_state
from dataclasses import dataclass
from typing import List, Any, Dict
from enum import Enum


@dataclass
class QuadrotorWaypointPlannerParams:
    waypoint_positions: np.ndarray # waypoints to visit
    waypoint_times: np.ndarray  # times to reach each waypoint

class QuadrotorWaypointPlanner(DiscreteTimeModel):
    def __init__(self, y0: Any, sample_rate: int, params: QuadrotorWaypointPlannerParams):
        """
        Initializes a QuadrotorWaypointPlanner object. Given some waypoints and times, the planner will
        switch between tracking the next desired waypoint based on the current time.

        Args:
            y0 (Any): The initial state of the quadrotor.
            sample_rate (int): The frequency (in Hz) at which the planner updates.
            params (QuadrotorWaypointPlannerParams): Configuration parameters including waypoint positions and times.
        """
        super().__init__(y0, sample_rate, 'dfb_planner', params, logging_level=LogLevel.INFO)
        self.b1d_prev = e1_N
        
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
        
        dynamics = self.input_models['quadrotor_state']
        r_b0_N, _, _, _ = decompose_state(dynamics.y)
        
        t_verified = np.clip(t, self.params.waypoint_times[0], self.params.waypoint_times[-1])
        idx = np.searchsorted(self.params.waypoint_times, t_verified, side='left')
        
        planar_projection = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 0]])
        
        yaw_des = self.params.waypoint_positions[idx] - r_b0_N
        
        if np.linalg.norm(yaw_des) < 0.1:
            b1d = self.b1d_prev
        else:
            b1d = planar_projection @ (yaw_des / np.linalg.norm(yaw_des))
        
        self.b1d_prev = b1d
        
        return [self.params.waypoint_positions[idx], np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)], \
               [b1d, np.zeros(3), np.zeros(3)]
        