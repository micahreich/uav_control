import bisect
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import sympy as sym
from hybrid_ode_sim.simulation.base import (ContinuousTimeModel,
                                            DiscreteTimeModel)
from hybrid_ode_sim.utils.logging_tools import LogLevel
from scipy.interpolate import interp1d
from spatialmath.base import (exp2r, q2r, qconj, qdotb, qnorm, qqmul, qslerp,
                              qunit, qvmul, r2q, skewa)

import uav_control.constants as constants
from uav_control.constants import (OMEGA_B0_B, OMEGA_B0_B_DIM, Q_NB, Q_NB_DIM,
                                   R_B0_N, R_B0_N_DIM, TAU_B0_B, TAU_B0_B_DIM,
                                   THRUST, THRUST_DIM, V_B0_N, V_B0_N_DIM,
                                   a_g_N, compose_state, compose_state_dot,
                                   decompose_control, decompose_state, e3,
                                   thrust_axis_B)
from uav_control.utils.math import (compute_cross_product_dot,
                                    compute_unit_vector_ddot,
                                    compute_unit_vector_dot, dxu_dx_jacobian,
                                    sym_Aq, sym_Gq, sym_H, sym_Lq, sym_Rq,
                                    sym_skewsym, vee)

nx = R_B0_N_DIM + V_B0_N_DIM + Q_NB_DIM + OMEGA_B0_B_DIM  # Number of states
nu = THRUST_DIM + TAU_B0_B_DIM  # Number of controls


@dataclass
class QuadrotorRigidBodyParams:
    m: float  # mass
    I: np.ndarray  # inertia matrix
    I_inv: np.ndarray = field(init=False)
    D_drag: np.ndarray  # diagonal drag coefficient matrix

    def __post_init__(self):
        self.I_inv = np.linalg.inv(self.I)


@dataclass
class QuadrotorLinearization:
    Jx: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nx)))  # A matrix
    Ju: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nu)))  # B matrix
    x0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx,)))  # linearization state
    u0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nu,)))  # linearization control


class QuadrotorRigidBodyDynamics(ContinuousTimeModel):
    def __init__(
        self,
        y0: np.ndarray,
        params: QuadrotorRigidBodyParams,
        name: str = "quadrotor_state",
        logging_level=LogLevel.ERROR,
        noise_y0: bool = False,
    ):
        """
        Initializes the QuadrotorDynamics class.

        Parameters
        ----------
        y0 : np.ndarray
            The initial state of the quadrotor. This should be a 13x1 vector with the following elements:
            - r_b0_N: 3x1 vector, the position of the quadrotor in the ENU frame
            - q_NB: 4x1 vector, the attitude of the quadrotor as a (unit) quaternion
            - v_b0_N: 3x1 vector, the velocity of the quadrotor in the ENU frame
            - omega_b0_B: 3x1 vector, the angular velocity of the quadrotor in the body frame
        params : QuadrotorRigidBodyParams
            The parameters of the UAV rigid body model.
        logging_level : LogLevel, optional
            The logging level for the model. Defaults to LogLevel.ERROR.
        noise_y0 : bool, optional
            Whether to add noise to the initial state. Sometimes required when using LQR to avoid exact 0s
            in some spots which make numerical solvers break down. Defaults to False.
        """
        if noise_y0:
            _sigma = 1e-4
            r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(y0)

            r_noised = r_b0_N + np.random.normal(0, _sigma, 3)
            q_noised = qqmul(q_NB, r2q(exp2r(np.random.normal(0, _sigma, 3))))
            v_noised = v_b0_N + np.random.normal(0, _sigma, 3)
            omega_noised = omega_b0_B + np.random.normal(0, _sigma, 3)

            y0 = compose_state(r_noised, q_noised, v_noised, omega_noised)

        super().__init__(y0, name, params, logging_level=logging_level)

    def output_validate(self, y: np.ndarray) -> np.ndarray:
        """
        Validates the output of the quadrotor dynamics, ensuring unit-norm quaternions.

        Parameters
        ----------
        y : np.ndarray
            The output to be validated.

        Returns
        -------
        np.ndarray
            The validated output.
        """

        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(y)

        return compose_state(r_b0_N, q_NB / qnorm(q_NB), v_b0_N, omega_b0_B)

    def continuous_dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculates the continuous dynamics of the quadrotor.

        Parameters
        ----------
        t : float
            The current time.
        y : np.ndarray
            The state vector of the quadrotor.

        Returns
        -------
        np.ndarray
            The derivative of the state vector.

        Notes
        -----
        Computes the rigid body dynamics (state derivatives) of a quadrotor UAV based on a first-order drag model.
        """

        try:
            allocated_wrench = self.input_models["allocator"].y
        except KeyError:
            self.logger.warning(
                "No input model found for 'allocator'. Defaulting to zero thrust and torque."
            )
            allocated_wrench = np.zeros(4)

        decomposed_state = decompose_state(y)
        collective_thrust, torque = decompose_control(allocated_wrench)

        return compose_state_dot(
            v_b0_N=self.compute_v_b0_N(decomposed_state),
            q_NB_dot=self.compute_q_NB_dot(decomposed_state),
            a_b0_N=self.compute_a_b0_N(decomposed_state, collective_thrust),
            omega_b0_B_dot=self.compute_omega_b0_B_dot(decomposed_state, torque),
        )

    def history_interpolator(self, t):
        t_idx = bisect.bisect_left(self.t_history, t)
        i_prev = max(0, t_idx - 1)
        i_next = min(len(self.t_history) - 1, t_idx)

        if i_prev == i_next:
            return self.y_history[i_prev]

        scale_prev = (self.t_history[i_next] - t) / (
            self.t_history[i_next] - self.t_history[i_prev]
        )
        scale_next = 1.0 - scale_prev

        r_b0_N_prev, q_NB_prev, v_b0_N_prev, omega_b0_B_prev = decompose_state(
            self.y_history[i_prev]
        )
        r_b0_N_next, q_NB_next, v_b0_N_next, omega_b0_B_next = decompose_state(
            self.y_history[i_next]
        )

        return compose_state(
            r_b0_N=r_b0_N_prev * scale_prev + r_b0_N_next * scale_next,
            q_NB=qunit(qslerp(q_NB_prev, q_NB_next, s=scale_next, shortest=True)),
            v_b0_N=v_b0_N_prev * scale_prev + v_b0_N_next * scale_next,
            omega_b0_B=omega_b0_B_prev * scale_prev + omega_b0_B_next * scale_next,
        )

    def compute_v_b0_N(self, decomposed_state):
        """
        Computes the velocity of the quadrotor in the ENU frame.

        Parameters
        ----------
        decomposed_state : tuple
            The decomposed state of the quadrotor.

        Returns
        -------
        np.ndarray
            The velocity of the quadrotor in the ENU frame.
        """
        _r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state

        return v_b0_N

    def compute_q_NB_dot(self, decomposed_state):
        """
        Computes the derivative of the quaternion representing the quadrotor's orientation.

        Parameters
        ----------
        decomposed_state : tuple
            The decomposed state of the quadrotor.

        Returns
        -------
        np.ndarray
            The derivative of the quaternion.
        """
        _r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state

        # Quaternion derivative with angular velocity expressed in the body frame
        return qdotb(q_NB, omega_b0_B)

    def compute_a_b0_N(self, decomposed_state, allocated_collective_thrust):
        """
        Computes the acceleration of the quadrotor in the ENU frame.

        Parameters
        ----------
        decomposed_state : tuple
            The decomposed state of the quadrotor.
        allocated_collective_thrust : float
            The allocated collective thrust.

        Returns
        -------
        np.ndarray
            The acceleration of the quadrotor in the ENU frame.
        """

        r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state

        rot_NB = q2r(q_NB)
        _, _, b3 = rot_NB.T

        # Force due to gravity
        f_g_N = self.params.m * a_g_N

        # Force due to rotors
        f_control_N = allocated_collective_thrust * b3

        # Force due to air resistance / drag
        f_drag_N = rot_NB @ self.params.D_drag @ rot_NB.T @ -v_b0_N

        a_b0_N = 1 / self.params.m * (f_g_N + f_control_N + f_drag_N)

        return a_b0_N

    def compute_j_b0_N(
        self,
        decomposed_state,
        allocated_collective_thrust,
        allocated_collective_thrust_dot,
    ):
        """
        Computes the jerk of the quadrotor in the ENU frame.

        Parameters
        ----------
        decomposed_state : tuple
            The decomposed state of the quadrotor.
        allocated_collective_thrust : float
            The allocated collective thrust.
        allocated_collective_thrust_dot : float
            The derivative of the allocated collective thrust.

        Returns
        -------
        np.ndarray
            The jerk of the quadrotor in the ENU frame.
        """

        r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state
        a_b0_N = self.compute_a_b0_N(decomposed_state, allocated_collective_thrust)

        rot_NB = q2r(q_NB)
        rot_NB_dot = rot_NB @ skewa(omega_b0_B)

        _, _, b3 = rot_NB.T
        _, _, b3_dot = rot_NB_dot.T

        f_control_N_dot = (
            allocated_collective_thrust_dot * b3 + allocated_collective_thrust * b3_dot
        )

        f_drag_N_dot = (
            rot_NB_dot @ self.params.D_drag @ rot_NB.T
            - rot_NB @ self.params.D_drag @ rot_NB.T @ rot_NB_dot @ rot_NB.T
        ) @ -v_b0_N + (rot_NB @ self.params.D_drag @ rot_NB.T) @ -a_b0_N

        j_b0_N = 1 / self.params.m * (f_control_N_dot + f_drag_N_dot)

        return j_b0_N

    def compute_omega_b0_B_dot(self, decomposed_state, allocated_torque_B):
        """
        Computes the derivative of the angular velocity of the quadrotor in the body frame.

        Parameters
        ----------
        decomposed_state : tuple
            The decomposed state of the quadrotor.
        allocated_torque_B : np.ndarray
            The allocated torque in the body frame.

        Returns
        -------
        np.ndarray
            The derivative of the angular velocity in the body frame.
        """

        _r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state

        # Euler's rotation equation for rigid bodies
        domega_b0_B = self.params.I_inv @ (
            allocated_torque_B - np.cross(omega_b0_B, self.params.I @ omega_b0_B)
        )

        return domega_b0_B

    def compute_differential_flatness_states_controls(
        self,
        r_b0_N_ref,
        v_b0_N_ref,
        a_b0_N_ref,
        j_b0_N_ref,
        s_b0_N_ref,
        b_1d,
        b_1d_dot,
        b_1d_ddot,
    ):
        """
        Compute the quadrotor state and controls as a function of the differentially flat
        outputs, ie position and desired first-body axis (and their time derivatives).

        Parameters
        ----------
        params : QuadrotorRigidBodyParams
            The parameters of the quadrotor rigid body model
        r_b0_N_ref : np.ndarray (3,)
            The reference position of the quadrotor CoM in the world frame
        v_b0_N_ref : np.ndarray (3,)
            The reference velocity of the quadrotor CoM in the world frame
        a_b0_N_ref : np.ndarray (3,)
            The reference acceleration of the quadrotor CoM in the world frame
        j_b0_N_ref : np.ndarray (3,)
            The reference jerk of the quadrotor CoM in the world frame
        s_b0_N_ref : np.ndarray (3,)
            The reference snap of the quadrotor CoM in the world frame
        b_1d : np.ndarray (3,)
            The desired first body-fixed axis of the quadrotor
        b_1d_dot : np.ndarray (3,)
            The time derivative of the desired first body-fixed axis of the quadrotor
        b_1d_ddot : np.ndarray (3,)
            The second time derivative of the desired first body-fixed axis of the quadrotor

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
            The quadrotor position, velocity, orientation (reference frame), and angular velocity (expressed in reference frame);
            The desired thrust vector and angular acceleration (expressed in reference frame)
        """

        a_b0_N_adjusted = a_b0_N_ref - a_g_N  # TODO: add drag model
        f_des = self.params.m * a_b0_N_adjusted  # Desired thrust vector, gravity-compensated

        # Construct desired/commanded orientation of the body frame
        b_3c = a_b0_N_adjusted / np.linalg.norm(a_b0_N_adjusted)  # Desired body z-axis

        C = np.cross(b_3c, b_1d)
        b_2c = C / np.linalg.norm(C)

        b_1c = np.cross(b_2c, b_3c)

        rot_ND = np.column_stack((b_1c, b_2c, b_3c))  # Rotation from desired frame to world

        # Construct desired/commanded angular velocity of the body frame
        b_3c_dot = compute_unit_vector_dot(a_b0_N_adjusted, j_b0_N_ref)

        C_dot = compute_cross_product_dot(b_3c, b_3c_dot, b_1d, b_1d_dot)
        b_2c_dot = compute_unit_vector_dot(C, C_dot)

        b_1c_dot = np.cross(b_2c_dot, b_3c) + np.cross(b_2c, b_3c_dot)

        rot_ND_dot = np.column_stack((b_1c_dot, b_2c_dot, b_3c_dot))
        omega_d0_D_skewa = rot_ND.T @ rot_ND_dot
        omega_d0_D = vee(omega_d0_D_skewa)

        # Construct desired/commanded angular acceleration of the body frame
        b_3c_ddot = compute_unit_vector_ddot(a_b0_N_adjusted, j_b0_N_ref, s_b0_N_ref)

        C_ddot = compute_cross_product_dot(
            b_3c_dot, b_3c_ddot, b_1d, b_1d_dot
        ) + compute_cross_product_dot(b_3c, b_3c_dot, b_1d_dot, b_1d_ddot)

        b_2c_ddot = compute_unit_vector_ddot(C, C_dot, C_ddot)

        b_1c_ddot = compute_cross_product_dot(
            b_2c_dot, b_2c_ddot, b_3c, b_3c_dot
        ) + compute_cross_product_dot(b_2c, b_2c_dot, b_3c_dot, b_3c_ddot)

        rot_ND_ddot = np.column_stack((b_1c_ddot, b_2c_ddot, b_3c_ddot))
        omega_d0_D_dot = vee(rot_ND.T @ rot_ND_ddot - omega_d0_D_skewa @ omega_d0_D_skewa)

        # Compute desired/commanded torque (expressed in the desired frame)
        tau_d0_D = self.params.I @ omega_d0_D_dot + np.cross(omega_d0_D, self.params.I @ omega_d0_D)

        return [r_b0_N_ref, v_b0_N_ref, rot_ND, omega_d0_D], [f_des, tau_d0_D]
