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
from uav_control.utils.math import compute_unit_vector_dot, compute_unit_vector_ddot


class ParametricCurve:
    def __init__(self, duration):
        self.duration = duration

    def x(self, t: float): raise NotImplementedError
    def vv(self, t: float): raise NotImplementedError
    def a(self, t: float): raise NotImplementedError
    def j(self, t: float): raise NotImplementedError
    def s(self, t: float): raise NotImplementedError

class ParametricCircle(ParametricCurve):
    def __init__(self, duration: float, radius: float, center: np.ndarray, normal: np.ndarray):
        super().__init__(duration)

        self.radius = radius
        self.center = center
        self.normal = normal / np.linalg.norm(normal)

        if np.allclose(self.normal, e1):
            a = e2
        else:
            a = e1

        self.uu = a - np.dot(a, self.normal) * self.normal
        self.uu /= np.linalg.norm(self.uu)

        self.vv = np.cross(self.normal, self.uu)

    def x(self, t: float):
        theta = 2 * np.pi * t / self.duration
        return self.center + self.radius * (np.cos(theta) * self.uu + np.sin(theta) * self.vv)

    def v(self, t: float):
        theta = 2 * np.pi * t / self.duration
        dtheta_dt = 2 * np.pi / self.duration
        return self.radius * (-np.sin(theta) * dtheta_dt * self.uu + np.cos(theta) * dtheta_dt * self.vv)

    def a(self, t: float):
        theta = 2 * np.pi * t / self.duration
        dtheta_dt = 2 * np.pi / self.duration
        dtheta_dt2 = dtheta_dt ** 2
        return self.radius * (-np.cos(theta) * dtheta_dt2 * self.uu - np.sin(theta) * dtheta_dt2 * self.vv)

    def j(self, t: float):
        theta = 2 * np.pi * t / self.duration
        dtheta_dt = 2 * np.pi / self.duration
        dtheta_dt3 = dtheta_dt ** 3
        return self.radius * (np.sin(theta) * dtheta_dt3 * self.uu - np.cos(theta) * dtheta_dt3 * self.vv)

    def s(self, t: float):
        theta = 2 * np.pi * t / self.duration
        dtheta_dt = 2 * np.pi / self.duration
        dtheta_dt4 = dtheta_dt ** 4
        return self.radius * (np.cos(theta) * dtheta_dt4 * self.uu + np.sin(theta) * dtheta_dt4 * self.vv)



class QuadrotorParametricCurvePlanner(DiscreteTimeModel):
    def __init__(
        self,
        y0: Any,
        sample_rate: int,
        curve: ParametricCurve,
        name: str = "parametric_planner",
        logging_level: LogLevel = LogLevel.ERROR,
        static_heading: bool = False,
        params=None,
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
        self.curve = curve
        self.static_heading = static_heading

        self.b1d_prev, self.b1d_dot_prev, self.b1d_ddot_prev = None, None, None

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

        if t <= self.curve.duration:
            x = self.curve.x(t)
            v = self.curve.v(t)
            a = self.curve.a(t)
            j = self.curve.j(t)
            s = self.curve.s(t)
        else:
            x = self.curve.x(self.curve.duration)
            v = np.zeros(3)
            a = np.zeros(3)
            j = np.zeros(3)
            s = np.zeros(3)

        if self.static_heading:
            b1d = np.array([1, 0, 0])
            b1d_dot = np.zeros(3)
            b1d_ddot = np.zeros(3)
        else:
            if np.allclose(v, np.zeros(3), atol=1e-6):
                b1d = self.b1d_prev
                b1d_dot = self.b1d_dot_prev
                b1d_ddot = self.b1d_ddot_prev
            else:
                b1d = v / np.linalg.norm(v)
                b1d_dot = compute_unit_vector_dot(v, a)
                b1d_ddot = compute_unit_vector_ddot(v, a, j)

        self.b1d_prev, self.b1d_dot_prev, self.b1d_ddot_prev = b1d, b1d_dot, b1d_ddot

        return [x, v, a, j, s], [b1d, b1d_dot, b1d_ddot]
