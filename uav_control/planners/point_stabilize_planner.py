from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import spatialmath as sm
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from spatialmath.base import (qconj, qdotb, qnorm, qvmul, rotx, roty, rotz,
                              skewa)

from uav_control.constants import R_B0_N, V_B0_N, decompose_state, e1_N, e2_N


@dataclass
class QuadrotorStabilizationPlannerParams:
    position: np.ndarray  # waypoints to visit
    b1d: float  # desired 1st-body-axis direction


class QuadrotorStabilizationPlanner(DiscreteTimeModel):
    def __init__(
        self, y0: Any, sample_rate: int, params: QuadrotorStabilizationPlannerParams
    ):
        """
        Initializes the QuadrotorStabilizationPlanner which is responsible for computing the desired states for point stabilization of a quadrotor.
        This planner uses the provided initial state, sample rate, and parameters to manage the stabilization process over discrete time steps.

        Args:
            y0 (Any): The initial state of the quadrotor.
            sample_rate (int): The frequency (in Hz) at which the planner updates.
            params (QuadrotorStabilizationPlannerParams): Configuration parameters including target positions and desired body-axis orientations.
        """
        super().__init__(
            y0, sample_rate, "dfb_planner", params, logging_level=LogLevel.INFO
        )
        self.b1d_prev = e1_N

    def discrete_dynamics(self, t: float, _y: Any) -> Any:
        """
        Returns the position and desired 1st-body-axis direction at the current time.

        Args:
            t (float): The current time.
            _y (Any): Not applicable.

        Returns:
            List[np.ndarray], List[np.ndarray]:
                - The desired position, velocity, acceleration, jerk, and snap.
                - The desired 1st-body-axis direction and its 1st, 2nd derivatives.
        """
        return [
            self.params.position,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        ], [self.params.b1d, np.zeros(3), np.zeros(3)]
