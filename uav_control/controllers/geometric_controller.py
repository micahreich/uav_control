from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from spatialmath.base import angvec2r, q2r, qconj, qqmul, qvmul, r2q, skewa

from uav_control.constants import a_g_N, compose_state_dot, decompose_state, e3, g, thrust_axis_B
from uav_control.dynamics import QuadrotorRigidBodyParams
from uav_control.utils.math import vee


@dataclass
class GeometricControllerParams:
    kP: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 proportional gains
    kD: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 derivative gains
    kR: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 attitude gains
    kOmega: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 angular velocity gains


@dataclass
class GeometricControllerTiltPrioritizedParams:
    kP: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 proportional gains
    kD: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 derivative gains
    kR_red: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 reduced attitude gains
    kR_yaw: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 yaw attitude gains
    kOmega: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 angular velocity gains


class GeometricController(DiscreteTimeModel):
    def __init__(
        self,
        y0: np.ndarray,
        sample_rate: int,
        params: Union[GeometricControllerParams, GeometricControllerTiltPrioritizedParams],
        rbd_params: QuadrotorRigidBodyParams,
        planner_name: str,
    ):
        """Initializes the GeometricController class.

        This controller is a differential flatness-based controller that tracks desired positions
        (and derivatives) and desired yaw angles (and derivatives).

        Based on the paper:
            Geometric Tracking Control of a Quadrotor UAV on SE(3)
            https://mathweb.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf

        Args:
            y0 (np.ndarray): The initial state of the controller; should be a 4x1 vector with the following elements:
                - collective_thrust: float, the collective thrust of the quadrotor
                - torques: 3x1 vector, the torques to be applied to the quadrotor
            sample_rate (int): The sample rate of the controller (Hz).
            params (GeometricControllerParams, optional): The parameters for the geometric controller.
                Defaults to GeometricControllerParams().
        """
        super().__init__(y0, sample_rate, "controller", params)
        self.rbd_params = rbd_params

        # Check if using tilt priority or not
        self.tilt_priority = isinstance(params, GeometricControllerTiltPrioritizedParams)
        self.f_app_prev = 9.8
        self.f_app_dot_prev = 0.0

        self.planner_name = planner_name

    def discrete_dynamics(self, _t: float, _y: np.ndarray) -> np.ndarray:
        """Calculate the control output of the geometric controller.

        Args:
            _t (float): Current time (not used in this function).
            _y (np.ndarray): The current state of the controller (not used in this function).

        Returns:
            np.ndarray: Desired control output of the geometric controller; a 4x1 vector with the following elements:
                - collective_thrust: float, the collective thrust of the quadrotor
                - torques: 3x1 vector, the desired torques to be applied to the quadrotor
        """
        dynamics = self.input_models["quadrotor_state"]
        decomposed_state = decompose_state(dynamics.y)

        [r_b0_N_ref, v_b0_N_ref, a_b0_N_ref, j_b0_N_ref, s_b0_N_ref], [
            b_1d,
            b_1d_dot,
            b_1d_ddot,
        ] = self.input_models[self.planner_name].y

        r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state
        rot_NB = q2r(q_NB)
        rot_NB_dot = rot_NB @ skewa(omega_b0_B)

        b_1, b_2, b_3 = rot_NB.T
        b_1_dot, b_2_dot, b_3_dot = rot_NB_dot.T

        # Note: errors for this controller are calculated like -kGain * (actual - reference), hence some things are flipped from convention

        # Construct desired acceleration vector, A, from position and velocity errors (and feedforward)
        e_x, e_v = r_b0_N - r_b0_N_ref, v_b0_N - v_b0_N_ref  # Position, velocity errors

        # fmt: off
        A = (
            1 / self.rbd_params.m * (
                -self.params.kP @ e_x
                + -self.params.kD @ e_v
                + self.rbd_params.m * a_b0_N_ref
            )
        )

        a_b0_N = dynamics.compute_a_b0_N(decomposed_state, self.f_app_prev)
        e_a = a_b0_N - a_b0_N_ref  # Acceleration error

        A_dot = (
            1 / self.rbd_params.m * (
                -self.params.kP @ e_v
                + -self.params.kD @ e_a
                + self.rbd_params.m * j_b0_N_ref
            )
        )

        j_b0_N = dynamics.compute_j_b0_N(
            decomposed_state, self.f_app_prev, self.f_app_dot_prev
        )
        e_j = j_b0_N - j_b0_N_ref

        A_ddot = (
            1 / self.rbd_params.m * (
                -self.params.kP @ e_a
                + -self.params.kD @ e_j
                + self.rbd_params.m * s_b0_N_ref
            )
        )
        # fmt: on

        [r_b0_N_ref, v_b0_N_ref, rot_ND, omega_d0_D], [
            f_cmd,
            omega_d0_D_dot,
        ] = dynamics.compute_differential_flatness_states_controls(
            r_b0_N_ref,
            v_b0_N_ref,
            A,
            A_dot,
            A_ddot,
            b_1d,
            b_1d_dot,
            b_1d_ddot,
        )

        # Project desired thrust vector onto current body z-axis to get desired collective thrust magnitude
        f_app = np.dot(f_cmd, b_3)

        self.f_app_prev = f_app
        self.f_app_dot_prev = np.dot(self.rbd_params.m * A_dot, b_3) + np.dot(f_cmd, b_3_dot)

        # Combine orientation, angular velocity errors + ff to get desired torque
        rot_BD = rot_NB.T @ rot_ND
        omega_d0_B = rot_BD @ omega_d0_D

        # Time derivative of the desired angular velocity in B frame, due to Transport Theorem
        omega_db_B = omega_d0_B - omega_b0_B
        omega_d0_B_dot = rot_BD @ omega_d0_D_dot + np.cross(omega_db_B, omega_d0_B)

        e_Omega = omega_b0_B - omega_d0_B

        # R_err represents the intrinsic rotation from the D frame to B frame, ie R_B = R_D * R_err
        rot_err = rot_ND.T @ rot_NB

        if self.tilt_priority:
            # First calculate reduced rotation error, ie rotation which brings D frame e3 axis to B frame e3 axis
            # by rotating about the axis e3_D x e3 (while making sure this axis is represented in the D frame)
            e3_D = rot_ND.T @ rot_NB[:, 2]
            n = np.cross(e3, e3_D)
            theta = np.arccos(np.dot(rot_ND[:, 2], rot_NB[:, 2]))

            if theta < 1e-6:
                axis = np.array([0, 0, 1])
                e_R_reduced_D = np.zeros(3)
            else:
                axis = n / np.linalg.norm(n)
                e_R_reduced_D = theta * axis

            rot_err_red_D = angvec2r(theta, axis, unit="rad")

            # Represent reduced error rotation in B frame
            e_R_reduced = rot_err.T @ e_R_reduced_D

            # Then calculate extrinsic rotation which brings D frame x axis to B frame x axis
            # by rotating about a fixed D-frame axis, ie R_B = R_{err,yaw} * (R_D * R_{err,reduced})
            rot_err_yaw_D = rot_NB @ (rot_ND @ rot_err_red_D).T
            e_R_yaw_D = 0.5 * vee(rot_err_yaw_D - rot_err_yaw_D.T)

            # Represent yaw error rotation in B frame
            e_R_yaw = rot_err.T @ e_R_yaw_D

            e_R_term = -self.params.kR_red @ e_R_reduced - self.params.kR_yaw @ e_R_yaw
        else:
            e_R = 0.5 * vee(rot_err - rot_err.T)
            e_R_term = -self.params.kR @ e_R

        M = (
            e_R_term
            + -self.params.kOmega @ e_Omega
            + np.cross(omega_b0_B, self.rbd_params.I @ omega_b0_B)
            + self.rbd_params.I @ omega_d0_B_dot
        )

        # Stack collective thrust and torques into control output
        control_output = np.hstack((f_app, M))
        return control_output
