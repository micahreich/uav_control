from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy as sp
import sympy as sym
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from numpy import cos, sin
from spatialmath.base import (
    angvec2r,
    angvelxform,
    exp2r,
    norm,
    q2r,
    qconj,
    qnorm,
    qqmul,
    qvmul,
    r2q,
    r2x,
    skew,
)

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
    g,
    thrust_axis_B,
)
from uav_control.dynamics import QuadrotorRigidBodyDynamics, QuadrotorRigidBodyParams
from uav_control.utils.math import (
    aa_to_dcm,
    compute_cross_product_dot,
    compute_unit_vector_ddot,
    compute_unit_vector_dot,
    dxnorm_dx_jacobian,
    dxu_dx_jacobian,
    qu_to_aa,
    qu_to_rodgigues_params,
    sym_Aq,
    sym_Gq,
    sym_H,
    sym_Lq,
    sym_Rq,
    sym_skewsym,
    vee,
)

nx, nu = 12, 4


@dataclass
class LQRControllerParams:
    Q: np.ndarray = field(default_factory=lambda: np.eye(12))  # nx x nx state cost matrix
    R: np.ndarray = field(default_factory=lambda: np.eye(4))  # nu x nu control cost matrix
    R_inv: np.ndarray = field(init=False)

    def __post_init__(self):
        if np.allclose(np.linalg.det(self.R), 0):
            raise ValueError("LQR gain parameter R must be invertible")

        self.R_inv = np.linalg.inv(self.R)

    @staticmethod
    def from_weights(
        r_weights: np.ndarray = np.ones(3),
        aa_weights: np.ndarray = np.ones(3),
        v_weights: np.ndarray = np.ones(3),
        omega_weights: np.ndarray = np.ones(3),
        collective_thrust_weight: float = 1.0,
        torque_weights: np.ndarray = np.ones(3),
    ):
        Q = np.diag(np.concatenate([r_weights, aa_weights, v_weights, omega_weights]))
        R = np.diag(np.concatenate([np.array([collective_thrust_weight]), torque_weights]))
        return LQRControllerParams(Q=Q, R=R)


@dataclass
class LQRGainSolverState:
    Jx: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nx)))  # A matrix
    Ju: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nu)))  # B matrix
    x0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx,)))  # linearization state
    u0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nu,)))  # linearization control
    K: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nu, nx)))  # LQR gain matrix


class LQRDynamicsLinearization(DiscreteTimeModel):
    def __init__(
        self,
        sample_rate: int,
        params: LQRControllerParams,
        rbd_params: QuadrotorRigidBodyParams,
        y0=None,
    ):
        super().__init__(y0, sample_rate, "lqr_linearization", params)
        self.rbd_params = rbd_params

    def _linearize_dynamics(self, x0, u0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearizes the dynamics of the UAV around the operating point (x0, u0).
        This function computes the Jacobian matrices A and B which represent the
        linearized system dynamics around the given state x0 and control input u0.
        The state x is decomposed into position, orientation (quaternion), velocity,
        and angular velocity. The control input u is decomposed into collective thrust
        and torque.

        Parameters
        ----------
        x0 : np.ndarray
            The state vector at the operating point, containing position, orientation
            (quaternion), velocity, and angular velocity.
        u0 : np.ndarray
            The control input vector at the operating point, containing collective
            thrust and torque.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - A (np.ndarray): The Jacobian matrix of the state dynamics with respect
              to the state vector (minimal coordinates, axis-angle orientation representation).
            - B (np.ndarray): The Jacobian matrix of the state dynamics with respect
              to the control input vector.
        """

        AA_DIM = 3

        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(x0)
        collective_thrust, tau = decompose_control(u0)
        rot_NB = q2r(q_NB)

        # We write the dynamics in the form xdot = f(x, u) and linearize around the operating point (x0, u0)
        # To avoid the issues that come with the unit quaternion being a redundant repsentation, we write the dynamics
        # using axis-angle representation for the attitude part of the state

        # fmt: off

        # Compute derivative of \dot{r} with respect to states
        drdot_dr = np.zeros((R_B0_N_DIM, R_B0_N_DIM))
        drdot_daa = np.zeros((R_B0_N_DIM, AA_DIM))
        drdot_dv = np.eye(R_B0_N_DIM)
        drdor_domega= np.zeros((R_B0_N_DIM, OMEGA_B0_B_DIM))

        # Compute derivative of \dot{aa} with respect to states
        daadot_dr = np.zeros((AA_DIM, R_B0_N_DIM))
        daadot_daa = np.zeros((AA_DIM, AA_DIM))
        daadot_dv = np.zeros((AA_DIM, V_B0_N_DIM))
        daa_domega = np.eye(AA_DIM)  # This is an approximation which holds for small angles

        # Compute derivative of \dot{v} with respect to states
        dvdot_dr = np.zeros((V_B0_N_DIM, R_B0_N_DIM))

        # See A compact formula for the derivative of a 3-D rotation in exponential coordinate
        # https://arxiv.org/pdf/1312.0788v1

        aa = r2x(rot_NB, representation="exp")
        aa_norm = norm(aa)

        c_B = collective_thrust * e3

        # d(c_W)/d(phi) = d(R_NB(v) @ c_B)/d(v)
        if aa_norm < 1e-8:
            dc_W_daa = -skew(c_B)
        else:
            dc_W_daa = -rot_NB @ skew(c_B) @ (np.outer(aa, aa) + (rot_NB.T - np.eye(3)) @ skew(aa)) / aa_norm**2

        dvdot_daa = 1/self.rbd_params.m * dc_W_daa

        dvdot_dv = np.zeros((V_B0_N_DIM, V_B0_N_DIM))  # TODO: implement this jacobian for linear drag
        dvdot_domega = np.zeros((V_B0_N_DIM, OMEGA_B0_B_DIM))

        # Compute derivative of \dot{omega} with respect to states
        domegadot_dr = np.zeros((OMEGA_B0_B_DIM, R_B0_N_DIM))
        domegadot_daa = np.zeros((OMEGA_B0_B_DIM, AA_DIM))
        domegadot_dv = np.zeros((OMEGA_B0_B_DIM, V_B0_N_DIM))
        domegadot_domega = self.rbd_params.I_inv @ (
            skew(self.rbd_params.I @ omega_b0_B) - skew(omega_b0_B) @ self.rbd_params.I
        )

        A = np.block([
            [drdot_dr, drdot_daa, drdot_dv, drdor_domega],
            [daadot_dr, daadot_daa, daadot_dv, daa_domega],
            [dvdot_dr, dvdot_daa, dvdot_dv, dvdot_domega],
            [domegadot_dr, domegadot_daa, domegadot_dv, domegadot_domega]
        ])

        # Compute derivative of \dot{r} with respect to controls
        drdot_dc = np.zeros((R_B0_N_DIM, THRUST_DIM))
        drdot_dtau = np.zeros((R_B0_N_DIM, TAU_B0_B_DIM))

        # Compute derivative of \dot{aa} with respect to controls
        daadot_dc = np.zeros((AA_DIM, THRUST_DIM))
        daadot_dtau = np.zeros((AA_DIM, TAU_B0_B_DIM))

        # Compute derivative of \dot{v} with respect to controls
        dc_W_dc = rot_NB[:, 2].reshape((-1, 1))  # (3 x 1) jacobian
        dvdot_dc = 1/self.rbd_params.m * dc_W_dc

        dvdot_dtau = np.zeros((V_B0_N_DIM, TAU_B0_B_DIM))

        # Compute derivative of \dot{omega} with respect to controls
        domegadot_dc = np.zeros((OMEGA_B0_B_DIM, THRUST_DIM))
        domegadot_dtau = self.rbd_params.I_inv

        B = np.block([
            [drdot_dc, drdot_dtau],
            [daadot_dc, daadot_dtau],
            [dvdot_dc, dvdot_dtau],
            [domegadot_dc, domegadot_dtau]
        ])

        assert A.shape == (nx, nx)
        assert B.shape == (nx, nu)

        # fmt: on

        return A, B

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        curr_state = self.input_models["quadrotor_state"].y
        curr_control = self.input_models["controller"].y

        A, B = self._linearize_dynamics(curr_state, curr_control)

        P = sp.linalg.solve_continuous_are(A, B, self.params.Q, self.params.R)
        K = self.params.R_inv @ B.T @ P

        assert K.shape == (4, 3 * 4)

        y = LQRGainSolverState(
            Jx=A,
            Ju=B,
            x0=curr_state,
            u0=curr_control,
            K=K,
        )

        return y


class LQRController(DiscreteTimeModel):
    def __init__(
        self,
        sample_rate: int,
        planner_name: str,
        y0=1e-2 * np.random.rand(4),
    ):
        super().__init__(y0, sample_rate, "controller")
        self.planner_name = planner_name

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        state_ref, control_ref = self.input_models[self.planner_name].y
        r_b0_N_ref, q_NB_ref, v_b0_N_ref, omega_b0_B_ref = decompose_state(state_ref)

        curr_state = self.input_models["quadrotor_state"].y
        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(curr_state)

        # Compute LQR controller output given by: u = u_reference - K @ del x
        lqr_gains: LQRGainSolverState = self.input_models["lqr_linearization"].y

        # fmt: off

        # We want the intrinsic rotation from the current orientation to the reference orientation
        # which would imply q_NB_ref = q_NB * q_err, or q_err = q_NB^-1 * q_NB_ref, but we need to
        # take the inverse of q_err due to the (x - x_ref) order of the LQR controller, meaning q_err = q_NB_ref^-1 * q_NB

        q_err = qqmul(qconj(q_NB_ref), q_NB)
        k_err, theta_err = qu_to_aa(q_err)

        x_err = np.concatenate([
            r_b0_N - r_b0_N_ref,
            k_err * theta_err,
            v_b0_N - v_b0_N_ref,
            omega_b0_B - omega_b0_B_ref,
        ])

        # fmt: on

        u_lqr = control_ref - lqr_gains.K @ x_err
        return u_lqr


# class LQRController(DiscreteTimeModel):
#     def __init__(
#         self,
#         y0: np.ndarray,
#         sample_rate: int,
#         rbd_params: QuadrotorRigidBodyParams,
#     ):
#         super().__init__(y0, sample_rate, "lqr_controller", None)
#         self.rbd_params = rbd_params

#     def discrete_dynamics(self, _t: float, _y: np.ndarray) -> np.ndarray:
#         # Desired state is [x, v, q], virtual input is omega
#         lqr_gains: LQRGainSolverState = self.input_models["lqr_gains"].y
#         curr_state = self.input_models["quadrotor_state"].y
#         _r_b0_N, _q_NB, _v_b0_N, omega_b0_B = decompose_state(curr_state)

#         # LQR controller output given by: u = u0 - K (x - x0)
#         u_lqr = lqr_gains.u0 - lqr_gains.K @ (LQRGainSolver.lqr_state(curr_state) - lqr_gains.x0)

#         # Low-level bodyrate controller to track desired angular velocities
#         collective_thrust_c, omega_c0_B = u_lqr[0], u_lqr[1:4]
#         control_wrench = self.rbd_params.I @ -self.params.kP @ (omega_b0_B - omega_c0_B) + np.cross(
#             omega_b0_B, self.rbd_params.I @ omega_b0_B
#         )

#         return u_lqr


# class BodyrateController(DiscreteTimeModel):
#     def __init__(
#         self,
#         y0: Any,
#         sample_rate: int,
#         params: BodyrateControllerParams,
#         rbd_params: QuadrotorRigidBodyParams,
#         logging_level=LogLevel.ERROR,
#     ):
#         super().__init__(y0, sample_rate, "bodyrate_controller", params, logging_level)
#         self.rbd_params = rbd_params

#     def discrete_dynamics(self, t: float, y: Any) -> Any:
#         lqr_control = self.input_models["lqr_controller"].y
#         curr_state = self.input_models["quadrotor_state"].y

#         _r_b0_N, _q_NB, _v_b0_N, omega_b0_B = decompose_state(curr_state)

#         collective_thrust_c, omega_c0_B = lqr_control[0], lqr_control[1:4]
#         omega_d0_B = -self.params.P @ (omega_b0_B - omega_c0_B)

#         # Low-level bodyrate controller to track desired angular velocities, feedback linearization
#         control_wrench = self.rbd_params.I @ omega_d0_B + np.cross(
#             omega_b0_B, self.rbd_params.I @ omega_b0_B
#         )
