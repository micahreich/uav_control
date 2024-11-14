import bisect
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import sympy as sym
from hybrid_ode_sim.simulation.base import ContinuousTimeModel, DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from scipy.interpolate import interp1d
from spatialmath.base import q2r, qconj, qdotb, qnorm, qslerp, qunit, qvmul, skewa

import uav_control.constants as constants
from uav_control.constants import (
    OMEGA_B0_B,
    OMEGA_B0_B_DIM,
    Q_NB,
    Q_NB_DIM,
    R_B0_N,
    R_B0_N_DIM,
    TAU_B0_B,
    TAU_B0_B_DIM,
    THRUST,
    THRUST_DIM,
    V_B0_N,
    V_B0_N_DIM,
    a_g_N,
    compose_state,
    compose_state_dot,
    decompose_control,
    decompose_state,
    e3,
    thrust_axis_B,
)
from uav_control.utils.math import (
    compute_cross_product_dot,
    compute_unit_vector_ddot,
    compute_unit_vector_dot,
    dqu_dq_jacobian,
    sym_Aq,
    sym_Gq,
    sym_H,
    sym_Lq,
    sym_Rq,
    sym_skewsym,
    vee,
)

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
    @staticmethod
    def _symbolic_dynamics() -> None:
        I = sym.MatrixSymbol('I', 3, 3)  # Inertia matrix
        I_inv = sym.MatrixSymbol('I_inv', 3, 3)  # Inertia matrix
        g = sym.symbols('g')  # Gravitation constant
        m = sym.symbols('m')  # Mass

        p = sym.MatrixSymbol('p', R_B0_N_DIM, 1)  # Position (world frame)
        v = sym.MatrixSymbol('v', V_B0_N_DIM, 1)  # Velocity (world frame)
        omega = sym.MatrixSymbol('omega', OMEGA_B0_B_DIM, 1)  # Angular velocity (body frame)
        q = sym.MatrixSymbol('q', Q_NB_DIM, 1)  # Attitude (unit quaternion)

        x = sym.BlockMatrix([[p], [q], [v], [omega]]).as_explicit()

        tau = sym.MatrixSymbol('tau', TAU_B0_B_DIM, 1)
        c = sym.symbols('c')

        u = sym.BlockMatrix([[sym.Matrix([c])], [tau]]).as_explicit()

        # Equations of motion
        pdot = v
        vdot = 1 / m * (sym.Matrix([0, 0, m * g]) + sym_Aq(q) @ sym.Matrix([0, 0, c]))
        qdot = 1 / 2 * sym_Lq(q) @ sym_H @ omega
        omegadot = I_inv @ (tau - sym_skewsym(omega) @ (I @ omega))

        dx_dt = sym.BlockMatrix([[pdot], [qdot], [vdot], [omegadot]]).as_explicit()

        return dx_dt, x, u

    dx_dt_symbolic, x, u = _symbolic_dynamics.__func__()

    def __init__(
        self,
        y0: np.ndarray,
        params: QuadrotorRigidBodyParams,
        logging_level=LogLevel.ERROR,
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
        """
        super().__init__(y0, "quadrotor_state", params, logging_level=logging_level)

        self.dx_dt_symbolic_paramified = self._paramify_symbolic_dynamics()

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

    def _paramify_symbolic_dynamics(self):
        I = sym.MatrixSymbol('I', 3, 3)  # Inertia matrix
        I_inv = sym.MatrixSymbol('I_inv', 3, 3)  # Inertia matrix
        g = sym.symbols('g')  # Gravitation constant
        m = sym.symbols('m')  # Mass

        parameter_substituions = {
            I: sym.Matrix(self.params.I),
            I_inv: sym.Matrix(self.params.I_inv),
            m: self.params.m,
            g: constants.g,
        }

        dx_dt_paramified = QuadrotorRigidBodyDynamics.dx_dt_symbolic.subs(parameter_substituions)
        dx_dt_symbolic_paramified = sym.lambdify(
            (QuadrotorRigidBodyDynamics.x, QuadrotorRigidBodyDynamics.u), dx_dt_paramified, 'numpy'
        )

        return dx_dt_symbolic_paramified

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

        return self.dx_dt_symbolic_paramified(y, allocated_wrench).flatten()

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

    def linearize(self, x0, u0) -> QuadrotorLinearization:
        """
        Linearizes the nonlinear dynamics of a quadrotor around a given state and control input.
        This function computes the Jacobian matrices of the system dynamics with respect to the state and control input,
        evaluated at the operating point (x0, u0). The resulting linearized system can be represented as:
        \dot{x} = J_x * x + J_u * u
        x0 : np.ndarray
            The state vector at the operating point, typically including position, orientation (quaternion), velocity, and angular velocity.
        u0 : np.ndarray
            The control input vector at the operating point, typically including thrust and torque.

        Parameters
        ----------
        x0 : np.ndarray (nx,)
            Linearization state
        u0 : np.ndarray (nu,)
            Linearization control

        Returns
        -------
        QuadrotorLinearization
            An object containing the Jacobian matrices J_x and J_u, and the operating point (x0, u0).
        """
        c, tau_b0_B = decompose_control(u0)
        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(x0)
        qw, qx, qy, qz = q_NB
        wx, wy, wz = omega_b0_B
        [[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]] = self.params.I

        # Construct individual jacobian elements
        dqu_dq = dqu_dq_jacobian(q_NB)

        dpdot_dv = np.eye(3)

        dvdot_dq = (
            2
            * c
            * 1
            / self.params.m
            * np.array(
                [
                    [qy, qz, qw, qx],
                    [-qx, -qw, qz, qy],
                    [0, -2 * qx, -2 * qy, 0],  # why is this different in the paper..?
                ]
            )
            @ dqu_dq
        )

        dqdot_dq = (
            1
            / 2
            * np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])
            @ dqu_dq
        )

        dqdot_domega = (
            1 / 2 * np.array([[-qx, -qy, -qz], [qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw]])
        )

        domegadot_domega = -self.params.I_inv @ np.array(
            [
                [
                    -I_xy * wz + I_xz * wy,
                    I_xz * wx - I_yy * wz + 2 * I_yz * wy + I_zz * wz,
                    -I_xy * wx - I_yy * wy - 2 * I_yz * wz + I_zz * wy,
                ],
                [
                    I_xx * wz - 2 * I_xz * wx - I_yz * wy - I_zz * wz,
                    I_xy * wz - I_yz * wx,
                    I_xx * wx + I_xy * wy + 2 * I_xz * wz - I_zz * wx,
                ],
                [
                    -I_xx * wy + 2 * I_xy * wx + I_yy * wy + I_yz * wz,
                    -I_xx * wx - 2 * I_xy * wy - I_xz * wz + I_yy * wx,
                    -I_xz * wy + I_yz * wx,
                ],
            ]
        )

        dvdot_dc = (
            1
            / self.params.m
            * np.array(
                [
                    [2 * (qw * qy + qx * qz)],
                    [2 * (qy * qz - qw * qx)],
                    [qw**2 - qx**2 - qy**2 + qz**2],
                ]
            )
        )

        domegadot_dtau = self.params.I_inv

        # Construct jacobians J_x, J_u to represent the linearized system dynamics as \dot{x} = J_x * x + J_u * u
        Jx = np.zeros(shape=(nx, nx))
        Ju = np.zeros(shape=(nx, nu))

        # fmt: off
        Jx = np.block([
            [np.zeros((R_B0_N_DIM, R_B0_N_DIM)),     dpdot_dv,                     np.zeros((R_B0_N_DIM, Q_NB_DIM)),     np.zeros((R_B0_N_DIM, OMEGA_B0_B_DIM))], # d\dot{p}/d{p, v, q, omega}
            [np.zeros((Q_NB_DIM, R_B0_N_DIM)),     np.zeros((Q_NB_DIM, V_B0_N_DIM)),     dqdot_dq,                     dqdot_domega],                 # d\dot{q}/d{p, v, q, omega}
            [np.zeros((V_B0_N_DIM, R_B0_N_DIM)),     np.zeros((V_B0_N_DIM, V_B0_N_DIM)),     dvdot_dq,                     np.zeros((V_B0_N_DIM, OMEGA_B0_B_DIM))], # d\dot{v}/d{p, v, q, omega}
            [np.zeros((OMEGA_B0_B_DIM, R_B0_N_DIM)), np.zeros((OMEGA_B0_B_DIM, V_B0_N_DIM)), np.zeros((OMEGA_B0_B_DIM, Q_NB_DIM)), domegadot_domega],             # d\dot{omega}/d{p, v, q, omega}
        ])

        Ju = np.block([
            [np.zeros((R_B0_N_DIM, THRUST_DIM)),     np.zeros((R_B0_N_DIM, TAU_B0_B_DIM))],
            [np.zeros((Q_NB_DIM, THRUST_DIM)),     np.zeros((Q_NB_DIM, TAU_B0_B_DIM))],
            [dvdot_dc,                     np.zeros((V_B0_N_DIM, TAU_B0_B_DIM))],
            [np.zeros((OMEGA_B0_B_DIM, THRUST_DIM)), domegadot_dtau],
        ])
        # fmt: on

        return QuadrotorLinearization(Jx, Ju, x0, u0)

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
