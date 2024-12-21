from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy as sp
import sympy as sym
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from numpy import cos, sin
from spatialmath.base import (angvec2r, angvelxform, exp2r, norm, q2r, qconj,
                              qnorm, qqmul, qvmul, r2q, r2x, skew, rotvelxform)

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
        rbd_params: QuadrotorRigidBodyParams,
        y0=None,
        name: str = "lqr_linearization",
        planner_name: str = "dfb_planner",
    ):
        super().__init__(y0, sample_rate, name, params)
        self.rbd_params = rbd_params
        self.planner_name = planner_name

    @staticmethod
    def compute_reduced_error_state(x, x0, u, u0) -> np.ndarray:
        r_d0_W, q_WD, v_d0_W, omega_d0_B = decompose_state(x0)
        r_b0_W, q_WB, v_b0_W, omega_b0_B = decompose_state(x)

        c_D, _ = decompose_control(u0)
        c_B, _ = decompose_control(u)

        v_d0_B = qvmul(qconj(q_WB), v_d0_W)
        v_b0_B = qvmul(qconj(q_WB), v_b0_W)

        R_WD = q2r(q_WD)
        R_WB = q2r(q_WB)

        r_bar = r_b0_W - r_d0_W
        omega_bar = omega_b0_B - omega_d0_B
        v_bar = v_b0_B - v_d0_B
        phi_bar = r2x(R_WD.T @ R_WB, representation="exp")

        x_err = np.concatenate([r_bar, phi_bar, v_bar])
        u_err = np.concatenate([[c_B - c_D], omega_bar])

        return x_err, u_err

    @staticmethod
    def decompose_reduced_error_state(x_err, u_err) -> Tuple[np.ndarray, np.ndarray]:
        r_err, phi_err, v_err = x_err[:3], x_err[3:6], x_err[6:9]
        c_err, omega_err = u_err[:1], u_err[1:]

        return [r_err, phi_err, v_err], [c_err, omega_err]

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

        AA_DIM = 3

        r_b0_W, q_WB, v_b0_W, omega_b0_B = decompose_state(x)
        c_B, _ = decompose_control(u)

        v_b0_B = qvmul(qconj(q_WB), v_b0_W)

        R_WB = q2r(q_WB)

        x_err, u_err = self.compute_reduced_error_state(x, x0, u, u0)
        [r_err, phi_err, v_err], [c_err, omega_err] = self.decompose_reduced_error_state(x_err, u_err)

        dr_err_dot__dr_err = skew(R_WB @ omega_err)
        dr_err_dot__dphi_err = np.zeros((R_B0_N_DIM, AA_DIM))
        dr_err_dot__dv_err = R_WB
        dr_err_dot__dc_err = np.zeros((R_B0_N_DIM, THRUST_DIM))
        dr_err_dot__domega_err = -R_WB @ skew(R_WB.T @ r_err)

        dphi_err_dot__dr_err = np.zeros((AA_DIM, R_B0_N_DIM))
        dphi_err_dot__dphi_err = np.zeros((AA_DIM, AA_DIM))
        dphi_err_dot__dv_err = np.zeros((AA_DIM, V_B0_N_DIM))
        dphi_err_dot__dc_err = np.zeros((AA_DIM, THRUST_DIM))
        dphi_err_dot__domega_err = np.eye(AA_DIM)

        dv_err_dot__dr_err = np.zeros((V_B0_N_DIM, R_B0_N_DIM))
        dv_err_dot__dphi_err = 1/self.rbd_params.m * (-skew(e3 * c_B) + skew(e3 * c_err))
        dv_err_dot__dv_err = -skew(omega_b0_B) + skew(omega_err)
        dv_err_dot__dc_err = (1/self.rbd_params.m * (e3 - skew(phi_err) @ e3)).reshape((-1, 1))
        dv_err_dot__domega_err = skew(v_b0_B) -skew(v_err)

        A = np.block([
            [dr_err_dot__dr_err, dr_err_dot__dphi_err, dr_err_dot__dv_err],
            [dphi_err_dot__dr_err, dphi_err_dot__dphi_err, dphi_err_dot__dv_err],
            [dv_err_dot__dr_err, dv_err_dot__dphi_err, dv_err_dot__dv_err]
        ])

        B = np.block([
            [dr_err_dot__dc_err, dr_err_dot__domega_err],
            [dphi_err_dot__dc_err, dphi_err_dot__domega_err],
            [dv_err_dot__dc_err, dv_err_dot__domega_err]
        ])



        # # We write the dynamics in the form xdot = f(x, u) and linearize around the operating point (x0, u0)
        # # To avoid the issues that come with the unit quaternion being a redundant repsentation, we write the dynamics
        # # using axis-angle representation for the attitude part of the state

        # # fmt: off

        # # Compute derivative of \dot{r} with respect to states
        # drdot_dr = np.zeros((R_B0_N_DIM, R_B0_N_DIM))
        # drdot_daa = np.zeros((R_B0_N_DIM, AA_DIM))
        # drdot_dv = np.eye(R_B0_N_DIM)
        # drdor_domega= np.zeros((R_B0_N_DIM, OMEGA_B0_B_DIM))

        # # Compute derivative of \dot{aa} with respect to states
        # daadot_dr = np.zeros((AA_DIM, R_B0_N_DIM))
        # daadot_daa = np.zeros((AA_DIM, AA_DIM))
        # daadot_dv = np.zeros((AA_DIM, V_B0_N_DIM))
        # daa_domega = np.eye(AA_DIM) #rotvelxform(aa, inverse=True, representation="exp") @ rot_NB

        # #np.eye(AA_DIM)  # This is an approximation which holds for small angles

        # # Compute derivative of \dot{v} with respect to states
        # dvdot_dr = np.zeros((V_B0_N_DIM, R_B0_N_DIM))

        # # See A compact formula for the derivative of a 3-D rotation in exponential coordinate
        # # https://arxiv.org/pdf/1312.0788v1

        # c_B = collective_thrust * e3

        # dc_W_daa = -rot_NB @ skew(c_B) @ (np.outer(aa, aa) + (rot_NB.T - np.eye(3)) @ skew(aa)) / aa_norm**2

        # dvdot_daa = 1/self.rbd_params.m * dc_W_daa

        # dvdot_dv = np.zeros((V_B0_N_DIM, V_B0_N_DIM))  # TODO: implement this jacobian for linear drag
        # dvdot_domega = np.zeros((V_B0_N_DIM, OMEGA_B0_B_DIM))

        # # Compute derivative of \dot{omega} with respect to states
        # domegadot_dr = np.zeros((OMEGA_B0_B_DIM, R_B0_N_DIM))
        # domegadot_daa = np.zeros((OMEGA_B0_B_DIM, AA_DIM))
        # domegadot_dv = np.zeros((OMEGA_B0_B_DIM, V_B0_N_DIM))
        # domegadot_domega = self.rbd_params.I_inv @ (
        #     skew(self.rbd_params.I @ omega_b0_B) - skew(omega_b0_B) @ self.rbd_params.I
        # )

        # # print("domegadot_domega delta:", np.linalg.norm(domegadot_domega - ))

        # A = np.block([
        #     [drdot_dr, drdot_daa, drdot_dv, drdor_domega],
        #     [daadot_dr, daadot_daa, daadot_dv, daa_domega],
        #     [dvdot_dr, dvdot_daa, dvdot_dv, dvdot_domega],
        #     [domegadot_dr, domegadot_daa, domegadot_dv, domegadot_domega]
        # ])

        # if self.y is not None:
        #     A_prev = self.y.Jx
        #     A_diff = A - A_prev

        #     if np.linalg.norm(A_diff) > 10:
        #         print(f"A_diff ({np.linalg.norm(A_diff)}): ", A_diff[6:9, 3:6], np.linalg.norm(A_diff[6:9, 3:6]))
        #         print(f"aa_norm: {aa_norm}")

        #         x_diff = x0 - self.y.x0
        #         u_diff = u0 - self.y.u0

        #         print(f"x_diff: {np.linalg.norm(x_diff)}")
        #         print(f"u_diff: {u_diff}")

        # # Compute derivative of \dot{r} with respect to controls
        # drdot_dc = np.zeros((R_B0_N_DIM, THRUST_DIM))
        # drdot_dtau = np.zeros((R_B0_N_DIM, TAU_B0_B_DIM))

        # # Compute derivative of \dot{aa} with respect to controls
        # daadot_dc = np.zeros((AA_DIM, THRUST_DIM))
        # daadot_dtau = np.zeros((AA_DIM, TAU_B0_B_DIM))

        # # Compute derivative of \dot{v} with respect to controls
        # dc_W_dc = rot_NB[:, 2].reshape((-1, 1))  # (3 x 1) jacobian
        # dvdot_dc = 1/self.rbd_params.m * dc_W_dc

        # dvdot_dtau = np.zeros((V_B0_N_DIM, TAU_B0_B_DIM))

        # # Compute derivative of \dot{omega} with respect to controls
        # domegadot_dc = np.zeros((OMEGA_B0_B_DIM, THRUST_DIM))
        # domegadot_dtau = self.rbd_params.I_inv

        # B = np.block([
        #     [drdot_dc, drdot_dtau],
        #     [daadot_dc, daadot_dtau],
        #     [dvdot_dc, dvdot_dtau],
        #     [domegadot_dc, domegadot_dtau]
        # ])

        # assert A.shape == (nx, nx)
        # assert B.shape == (nx, nu)

        # fmt: on

        return A, B

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        curr_state = self.input_models["quadrotor_state"].y
        curr_control = self.input_models["controller"].y
        desired_state, desired_control = self.input_models[self.planner_name].y

        A, B = self._linearize_dynamics(curr_state, desired_state, curr_control, desired_control)

        P = sp.linalg.solve_continuous_are(A, B, self.params.Q, self.params.R)
        K = self.params.R_inv @ B.T @ P

        # if self.y is not None:
        #     Kdiff = K - self.y.K
        #     Adiff = A - self.y.Jx
        #     Bdiff = B - self.y.Ju

        #     # print("Kdiff: ", np.linalg.norm(Kdiff))
        #     # print("Adiff: ", np.linalg.norm(Adiff))
        #     # print("Bdiff: ", np.linalg.norm(Bdiff))

        # assert K.shape == (4, 3 * 4)

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

        x_err, _ = LQRDynamicsLinearization.compute_reduced_error_state(curr_state, lqr_gains.x0, y, lqr_gains.u0)

        reduced_lqr_u0 = np.concatenate([
            [lqr_gains.u0[0]], # c_D
            lqr_gains.x0[OMEGA_B0_B]
        ])

        u_lqr = reduced_lqr_u0 - lqr_gains.K @ x_err
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
        rbd_params: QuadrotorRigidBodyParams,
        logging_level=LogLevel.ERROR,
    ):
        super().__init__(y0, sample_rate, "controller", params, logging_level)
        self.rbd_params = rbd_params

    def discrete_dynamics(self, _t: float, _y: Any) -> Any:
        lqr_control = self.input_models["lqr_controller"].y
        curr_state = self.input_models["quadrotor_state"].y

        _r_b0_N, _q_NB, _v_b0_N, omega_b0_B = decompose_state(curr_state)

        collective_thrust_c, omega_c0_B = lqr_control[0], lqr_control[1:4]

        domega_b0_B = -self.params.P @ (omega_b0_B - omega_c0_B)

        # Low-level bodyrate controller to track desired angular velocities, feedback linearization
        control_wrench = self.rbd_params.I @ domega_b0_B + np.cross(
            omega_b0_B, self.rbd_params.I @ omega_b0_B
        )

        return np.concatenate([[collective_thrust_c], control_wrench])
