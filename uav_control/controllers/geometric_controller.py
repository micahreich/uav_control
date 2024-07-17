from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from spatialmath.base import q2r, qconj, qqmul, qvmul, r2q, skewa

from uav_control.constants import (a_g_N, compose_state_dot, decompose_state,
                                   e3_B, e3_N, g, thrust_axis_B)
from uav_control.dynamics import QuadrotorRigidBodyParams


@dataclass
class QuadrotorGeometricControllerParams:
    kP: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 proportional gains
    kD: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 derivative gains
    kR: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 attitude gains
    kOmega: np.ndarray = field(
        default_factory=lambda: np.eye(3)
    )  # 3x3 angular velocity gains
    m: float = 1.0  # mass of the quadrotor
    I: np.ndarray = field(
        default_factory=lambda: np.eye(3)
    )  # inertia matrix of the quadrotor
    # D_drag: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # drag matrix of the quadrotor


def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def compute_unit_vector_dot(u, u_dot):
    u_norm = np.linalg.norm(u)
    return u_dot / u_norm - np.dot(u, u_dot) / u_norm**3 * u


def compute_unit_vector_ddot(u, u_dot, u_ddot):
    u_norm = np.linalg.norm(u)
    u_dot_norm = np.dot(u, u_dot) / u_norm
    return (
        (1 / u_norm) * u_ddot
        - (2 * np.dot(u, u_dot) / u_norm**3) * u_dot
        - ((u_dot_norm**2 + np.dot(u, u_ddot)) / u_norm**3) * u
        + (3 * np.dot(u, u_dot) ** 2 / u_norm**5) * u
    )


def compute_cross_product_dot(u, u_dot, v, v_dot):
    return np.cross(u_dot, v) + np.cross(u, v_dot)


class QuadrotorGeometricController(DiscreteTimeModel):
    def __init__(
        self,
        y0: np.ndarray,
        sample_rate: int,
        params=QuadrotorGeometricControllerParams(),
    ):
        """Initializes the QuadrotorGeometricController class.

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
            params (QuadrotorGeometricControllerParams, optional): The parameters for the geometric controller.
                Defaults to QuadrotorGeometricControllerParams().
        """
        super().__init__(y0, sample_rate, "controller", params)

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
        ] = self.input_models["dfb_planner"].y

        r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state
        rot_NB = q2r(q_NB)
        rot_NB_dot = rot_NB @ skewa(omega_b0_B)

        b_1, b_2, b_3 = rot_NB.T
        b_1_dot, b_2_dot, b_3_dot = rot_NB_dot.T

        # Construct desired thrust vector
        e_x, e_v = r_b0_N - r_b0_N_ref, v_b0_N - v_b0_N_ref

        A = (
            -self.params.kP @ e_x
            + -self.params.kD @ e_v
            - self.params.m * a_g_N
            + self.params.m * a_b0_N_ref
        )

        # Project desired thrust vector onto current body z-axis to get desired collective thrust magnitude
        f = np.dot(A, b_3)

        # Construct desired/commanded orientation of the body frame
        b_3c = A / np.linalg.norm(A)  # Desired body z-axis

        C = np.cross(b_3c, b_1d)
        b_2c = C / np.linalg.norm(C)

        b_1c = np.cross(b_2c, b_3c)

        rot_ND = np.column_stack(
            (b_1c, b_2c, b_3c)
        )  # Rotation from desired frame to world

        # Construct desired/commanded angular velocity of the body frame
        a_b0_N = dynamics.compute_a_b0_N(decomposed_state, f)
        e_a = a_b0_N - a_b0_N_ref

        A_dot = (
            -self.params.kP @ e_v + -self.params.kD @ e_a + self.params.m * j_b0_N_ref
        )
        b_3c_dot = compute_unit_vector_dot(A, A_dot)

        C_dot = compute_cross_product_dot(b_3c, b_3c_dot, b_1d, b_1d_dot)
        b_2c_dot = compute_unit_vector_dot(C, C_dot)

        b_1c_dot = np.cross(b_2c_dot, b_3c) + np.cross(b_2c, b_3c_dot)

        rot_ND_dot = np.column_stack((b_1c_dot, b_2c_dot, b_3c_dot))
        omega_d0_D_skewa = rot_ND.T @ rot_ND_dot
        omega_d0_D = vee(omega_d0_D_skewa)

        # Construct desired/commanded angular acceleration of the body frame
        f_dot = np.dot(A_dot, b_3) + np.dot(A, b_3_dot)
        j_b0_N = dynamics.compute_j_b0_N(decomposed_state, f, f_dot)
        e_j = j_b0_N - j_b0_N_ref

        A_ddot = (
            -self.params.kP @ e_a + -self.params.kD @ e_j + self.params.m * s_b0_N_ref
        )

        b_3c_ddot = compute_unit_vector_ddot(A, A_dot, A_ddot)

        C_ddot = compute_cross_product_dot(
            b_3c_dot, b_3c_ddot, b_1d, b_1d_dot
        ) + compute_cross_product_dot(b_3c, b_3c_dot, b_1d_dot, b_1d_ddot)

        b_2c_ddot = compute_unit_vector_ddot(C, C_dot, C_ddot)

        b_1c_ddot = compute_cross_product_dot(
            b_2c_dot, b_2c_ddot, b_3c, b_3c_dot
        ) + compute_cross_product_dot(b_2c, b_2c_dot, b_3c_dot, b_3c_ddot)

        rot_ND_ddot = np.column_stack((b_1c_ddot, b_2c_ddot, b_3c_ddot))
        omega_d0_D_dot = vee(
            rot_ND.T @ rot_ND_ddot - omega_d0_D_skewa @ omega_d0_D_skewa
        )

        # Combine orientation, angular velocity errors + ff to get desired torque
        omega_d0_B = rot_NB.T @ rot_ND @ omega_d0_D
        omega_d0_B_dot = rot_NB.T @ rot_ND @ omega_d0_D_dot

        e_R = 0.5 * vee(rot_ND.T @ rot_NB - rot_NB.T @ rot_ND)
        e_Omega = omega_b0_B - omega_d0_B

        M = (
            -self.params.kR @ e_R
            + -self.params.kOmega @ e_Omega
            + np.cross(omega_b0_B, self.params.I @ omega_b0_B)
            + self.params.I @ omega_d0_B_dot
        )
        # self.params.I @ np.cross(omega_b0_B, rot_NB.T @ rot_ND @ omega_d0_D) + \

        # Stack collective thrust and torques into control output
        control_output = np.hstack((f, M))
        return control_output
