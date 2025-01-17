from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy as sp
import sympy as sym
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from numpy import cos, sin
from spatialmath.base import (angvec2r, angvelxform, exp2r, norm, q2r, qconj,
                              qnorm, qqmul, qvmul, r2q, r2x, skew, rotvelxform,
                              trlog, vex)

import uav_control.constants as constants
from uav_control.constants import (OMEGA_B0_B, OMEGA_B0_B_DIM, Q_NB, Q_NB_DIM,
                                   R_B0_N, R_B0_N_DIM, TAU_B0_B, TAU_B0_B_DIM,
                                   THRUST, THRUST_DIM, V_B0_N, V_B0_N_DIM,
                                   a_g_N, compose_state, compose_state_dot,
                                   decompose_control, decompose_state, e3, g,
                                   thrust_axis_B)
from uav_control.dynamics import (QuadrotorRigidBodyDynamics,
                                  QuadrotorRigidBodyParams)
from uav_control.utils.math import (aa_to_dcm, compute_cross_product_dot,
                                    compute_unit_vector_ddot,
                                    compute_unit_vector_dot,
                                    dxnorm_dx_jacobian, dxu_dx_jacobian,
                                    qu_to_aa, qu_to_rodgigues_params, sym_Aq,
                                    sym_Gq, sym_H, sym_Lq, sym_Rq, sym_skewsym,
                                    vee, qu_err_to_aa_err)

nx, nu = 9, 4


@dataclass
class LQRControllerParams:
    Q: np.ndarray = field(default_factory=lambda: np.eye(nx))  # nx x nx state cost matrix
    R: np.ndarray = field(default_factory=lambda: np.eye(nu))  # nu x nu control cost matrix
    R_inv: np.ndarray = field(init=False)

    def __post_init__(self):
        if np.allclose(np.linalg.det(self.R), 0):
            raise ValueError("LQR gain parameter R must be invertible")

        self.R_inv = np.linalg.inv(self.R)

    # @staticmethod
    # def from_weights(
    #     r_weights: np.ndarray = np.ones(3),
    #     aa_weights: np.ndarray = np.ones(3),
    #     v_weights: np.ndarray = np.ones(3),
    #     omega_weights: np.ndarray = np.ones(3),
    #     collective_thrust_weight: float = 1.0,
    #     torque_weights: np.ndarray = np.ones(3),
    # ):
    #     Q = np.diag(np.concatenate([r_weights, aa_weights, v_weights, omega_weights]))
    #     R = np.diag(np.concatenate([np.array([collective_thrust_weight]), torque_weights]))
    #     return LQRControllerParams(Q=Q, R=R)

    def from_weights(
        r_weights: np.ndarray = np.ones(3),
        aa_weights: np.ndarray = np.ones(3),
        v_weights: np.ndarray = np.ones(3),
        omega_weights: np.ndarray = np.ones(3),
        collective_thrust_weight: float = 1.0,
    ):
        Q = np.diag(np.concatenate([r_weights, aa_weights, v_weights]))
        R = np.diag(np.concatenate([np.array([collective_thrust_weight]), omega_weights]))
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
        y0=None,
        name: str = "lqr_linearization",
        planner_name: str = "dfb_planner",
    ):
        super().__init__(y0, sample_rate, name, params)
        self.planner_name = planner_name

    @staticmethod
    def compute_reduced_error_state(x, x0, u, u0) -> Tuple[np.ndarray, np.ndarray]:
        r_d0_W, q_WD, v_d0_W, omega_d0_B = decompose_state(x0)
        r_b0_W, q_WB, v_b0_W, omega_b0_B = decompose_state(x)

        c_D, _ = decompose_control(u0)
        c_B, _ = decompose_control(u)

        delta_p = r_b0_W - r_d0_W
        delta_R = q2r(qqmul(qconj(q_WD), q_WB))
        delta_theta = r2x(delta_R, representation="exp")
        delta_v = v_b0_W - v_d0_W

        delta_omega = omega_b0_B - omega_d0_B
        delta_c = c_B - c_D

        delta_x = np.concatenate([delta_p, delta_theta, delta_v])
        delta_u = np.concatenate([[delta_c], delta_omega])

        return delta_x, delta_u

    @staticmethod
    def decompose_reduced_error_state(x_err, u_err) -> Tuple[np.ndarray, np.ndarray]:
        p_err, theta_err, v_err = x_err[:3], x_err[3:6], x_err[6:9]
        c_err, omega_err = u_err[:1], u_err[1:]

        return [p_err, theta_err, v_err], [c_err, omega_err]

    def _linearize_dynamics(self, x, x0, u, u0) -> Tuple[np.ndarray, np.ndarray]:
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

        THETA_DIM = 3
        r_d0_W, q_WD, v_d0_W, omega_d0_B = decompose_state(x0)
        c_D, _ = decompose_control(u0)

        R_WD = q2r(q_WD)
        I = np.eye(3)

        dynamics = self.input_models["quadrotor_state"]
        m = dynamics.params.m

        [delta_p, delta_theta, delta_v], [delta_c, delta_omega] = \
            self.decompose_reduced_error_state(
                *self.compute_reduced_error_state(x, x0, u, u0)
            )

        # fmt: off

        dp_err_dot__dp_err = np.zeros(shape=(R_B0_N_DIM, R_B0_N_DIM))
        dp_err_dot__dtheta_err = np.zeros(shape=(R_B0_N_DIM, THETA_DIM))
        dp_err_dot__dv_err = I

        dtheta_err_dot__dp_err = np.zeros(shape=(THETA_DIM, R_B0_N_DIM))
        dtheta_err_dot__dtheta_err = -1/2 * skew(delta_omega)
        dtheta_err_dot__dv_err = np.zeros(shape=(THETA_DIM, V_B0_N_DIM))

        dv_err_dot__dp_err = np.zeros(shape=(V_B0_N_DIM, R_B0_N_DIM))
        dv_err_dot__dtheta_err = -1/m * R_WD @ skew(e3 * c_D + delta_c)
        dv_err_dot__dv_err = np.zeros(shape=(V_B0_N_DIM, V_B0_N_DIM))

        dp_err_dot__dc_err = np.zeros(shape=(R_B0_N_DIM, THRUST_DIM))
        dp_err_dot__domega_err = np.zeros(shape=(R_B0_N_DIM, OMEGA_B0_B_DIM))

        dtheta_err_dot__dc_err = np.zeros(shape=(THETA_DIM, THRUST_DIM))
        dtheta_err_dot__domega_err = I + 1/2 * skew(delta_theta)

        dv_err_dot__dc_err = (1/m * R_WD @ (I + skew(delta_theta)) @ e3).reshape((-1, 1))
        dv_err_dot__domega_err = np.zeros(shape=(V_B0_N_DIM, OMEGA_B0_B_DIM))

        A = np.block([
            [dp_err_dot__dp_err, dp_err_dot__dtheta_err, dp_err_dot__dv_err],
            [dtheta_err_dot__dp_err, dtheta_err_dot__dtheta_err, dtheta_err_dot__dv_err],
            [dv_err_dot__dp_err, dv_err_dot__dtheta_err, dv_err_dot__dv_err]
        ])

        B = np.block([
            [dp_err_dot__dc_err, dp_err_dot__domega_err],
            [dtheta_err_dot__dc_err, dtheta_err_dot__domega_err],
            [dv_err_dot__dc_err, dv_err_dot__domega_err]
        ])

        # fmt: on

        return A, B

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        curr_state = self.input_models["quadrotor_state"].y
        curr_control = self.input_models["controller"].y

        desired_state, desired_control = self.input_models[self.planner_name].y

        A, B = self._linearize_dynamics(curr_state, desired_state, curr_control, desired_control)
        P = sp.linalg.solve_continuous_are(A, B, self.params.Q, self.params.R)
        K = self.params.R_inv @ B.T @ P

        y = LQRGainSolverState(
            Jx=A,
            Ju=B,
            x0=desired_state,
            u0=desired_control,
            K=K,
        )

        return y


class LQRController(DiscreteTimeModel):
    def __init__(
        self,
        sample_rate: int,
        y0=1e-2 * np.random.rand(4),
    ):
        super().__init__(y0, sample_rate, "lqr_controller")

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        curr_state = self.input_models["quadrotor_state"].y
        lqr_gains: LQRGainSolverState = self.input_models["lqr_linearization"].y
        nominal_control = lqr_gains.u0

        curr_delta_x, _ = LQRDynamicsLinearization.compute_reduced_error_state(curr_state, lqr_gains.x0, y, lqr_gains.u0)

        r_d0_W, q_WD, v_d0_W, omega_d0_B = decompose_state(lqr_gains.x0)
        c_D, _ = decompose_control(lqr_gains.u0)
        nominal_control = np.concatenate([[c_D], omega_d0_B])

        delta_u_lqr = -lqr_gains.K @ curr_delta_x
        u_lqr = delta_u_lqr + nominal_control

        return u_lqr


        # curr_state = self.input_models["quadrotor_state"].y
        # r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(curr_state)

        # # Compute LQR controller output given by: u = u_reference - K @ del x

        # # fmt: off

        # # We want the intrinsic rotation from the current orientation to the reference orientation
        # # which would imply q_NB_ref = q_NB * q_err, or q_err = q_NB^-1 * q_NB_ref, but we need to
        # # take the inverse of q_err due to the (x - x_ref) order of the LQR controller, meaning q_err = q_NB_ref^-1 * q_NB

        # k_err, theta_err = qu_err_to_aa_err(q_NB, q_NB_ref, curr_minus_ref=True)

        # x_err = np.concatenate([
        #     r_b0_N - r_b0_N_ref,
        #     k_err * theta_err,
        #     v_b0_N - v_b0_N_ref,
        #     omega_b0_B - omega_b0_B_ref,
        # ])

        # # print("x_err: ", x_err[:3], x_err[3:6])

        # # fmt: on

        # u_lqr = control_ref - lqr_gains.K @ x_err
        # self.y = u_lqr

        # return u_lqr


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


@dataclass
class BodyrateControllerParams:
    P: np.ndarray = field(default_factory=lambda: np.eye(3))

class BodyrateController(DiscreteTimeModel):
    def __init__(
        self,
        y0: Any,
        sample_rate: int,
        params: BodyrateControllerParams,
        logging_level=LogLevel.ERROR,
    ):
        super().__init__(y0, sample_rate, "controller", params, logging_level)

    def discrete_dynamics(self, _t: float, _y: Any) -> Any:
        lqr_control = self.input_models["lqr_controller"].y
        dynamics = self.input_models["quadrotor_state"]

        curr_state = dynamics.y
        _r_b0_N, _q_NB, _v_b0_N, omega_b0_B = decompose_state(curr_state)

        collective_thrust_c, omega_c0_B = lqr_control[0], lqr_control[1:4]

        omega_c0_B_dot = -self.params.P @ (omega_b0_B - omega_c0_B)

        # Low-level bodyrate controller to track desired angular velocities, feedback linearization
        control_wrench = dynamics.params.I @ omega_c0_B_dot + np.cross(
            omega_b0_B, dynamics.params.I @ omega_b0_B
        )

        return np.concatenate([[collective_thrust_c], control_wrench])
