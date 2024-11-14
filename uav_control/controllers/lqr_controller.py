from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy as sp
import sympy as sym
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from spatialmath.base import angvec2r, q2r, qconj, qnorm, qqmul, qvmul, r2q, skewa

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
    compute_cross_product_dot,
    compute_unit_vector_ddot,
    compute_unit_vector_dot,
    dqu_dq_jacobian,
    qu_to_rodgigues_params,
    sym_Aq,
    sym_Gq,
    sym_H,
    sym_Lq,
    sym_Rq,
    sym_skewsym,
    vee,
)

nx, nu = 3 + 4 + 3 + 3, 1 + 3


@dataclass
class LQRControllerParams:
    Q: np.ndarray = field(default_factory=lambda: np.eye(nx))  # nx x nx state cost matrix
    R: np.ndarray = field(default_factory=lambda: np.eye(nu))  # nu x nu control cost matrix
    R_inv: np.ndarray = field(init=False)

    def __post_init__(self):
        if np.allclose(np.linalg.det(self.params.R), 0):
            raise ValueError("LQR gain parameter R must be invertible")

        self.R_inv = np.linalg.inv(self.params.R)


@dataclass
class LQRGainSolverState:
    Jx: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nx)))  # A matrix
    Ju: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nu)))  # B matrix
    x0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx,)))  # linearization state
    u0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nu,)))  # linearization control
    K: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nu, nx)))  # LQR gain matrix


class LQRController(DiscreteTimeModel):
    def __init__(
        self,
        y0: LQRGainSolverState,
        sample_rate: int,
        params: LQRControllerParams,
        rbd_params: QuadrotorRigidBodyParams,
    ):
        # https://underactuated.mit.edu/lqr.html
        super().__init__(y0, sample_rate, "lqr_controller", params)
        self.rbd_params = rbd_params

        (
            self.df_dx_symbolic_paramified,
            self.df_du_symbolic_paramified,
        ) = self._dynamics_jacobians_np()

    def _dynamics_jacobians_np(self):
        df_dxbar = QuadrotorRigidBodyDynamics.dx_dt.jacobian(QuadrotorRigidBodyDynamics.x)
        attitude_jac_dfdx = sym.BlockDiagMatrix(
            sym.eye(R_B0_N_DIM),
            sym_Gq(QuadrotorRigidBodyDynamics.x[Q_NB, 0]),
            sym.eye(V_B0_N_DIM),
            sym.eye(OMEGA_B0_B_DIM),
        ).as_explicit()

        df_dx_symbolic = df_dxbar @ attitude_jac_dfdx  # Shape: (13, 12)
        df_du_symbolic = QuadrotorRigidBodyDynamics.dx_dt.jacobian(
            QuadrotorRigidBodyDynamics.u
        )  # Shape: (13, 4)

        I = sym.MatrixSymbol('I', 3, 3)  # Inertia matrix
        I_inv = sym.MatrixSymbol('I_inv', 3, 3)  # Inertia matrix
        g = sym.symbols('g')  # Gravitation constant
        m = sym.symbols('m')  # Mass

        parameter_substituions = {
            I: sym.Matrix(self.rbd_params.I),
            I_inv: sym.Matrix(self.rbd_params.I_inv),
            m: self.rbd_params.m,
            g: constants.g,
        }

        df_dx_paramified = df_dx_symbolic.subs(parameter_substituions)
        df_du_paramified = df_du_symbolic.subs(parameter_substituions)

        df_dx_symbolic_paramified = sym.lambdify(
            (QuadrotorRigidBodyDynamics.x, QuadrotorRigidBodyDynamics.u), df_dx_paramified, 'numpy'
        )

        df_du_symbolic_paramified = sym.lambdify(
            (QuadrotorRigidBodyDynamics.x, QuadrotorRigidBodyDynamics.u), df_du_paramified, 'numpy'
        )

        return df_dx_symbolic_paramified, df_du_symbolic_paramified

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        # Want to drive the system xerr = x - x_ref to zero, where error dynamics are given by
        # x_err_dot = f(x, u) - f(x_ref, u_ref), approximated as x_err_dot = f(x0, u0) + Jx (x - x0) + Ju (u - u0) - f(x_ref, u_ref)
        # Since we are linearizing around the reference state, control, we can approximate the error dynamics as
        # x_err_dot = Jx (x - x0) + Ju (u - u0)

        state_ref, control_ref = self.input_models["dfb_planner"].y

        r_b0_N_ref, q_NB_ref, v_b0_N_ref, omega_b0_B_ref = decompose_state(state_ref)

        # Compute LQR infinite horizon gain matrix K
        Jx = self.df_dx_symbolic_paramified(state_ref, control_ref)
        Ju = self.df_du_symbolic_paramified(state_ref, control_ref)

        P = sp.linalg.solve_continuous_are(Jx, Ju, self.params.Q, self.params.R)
        K = self.R_inv @ Ju.T @ P

        # Compute LQR controller output given by: u = u_reference - K @ del x
        curr_state = self.input_models["quadrotor_state"].y
        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(curr_state)

        q_err = qqmul(qconj(q_NB), q_NB_ref)

        x_err = np.concatenate(
            [
                r_b0_N - r_b0_N_ref,
                qu_to_rodgigues_params(q_err),
                v_b0_N - v_b0_N_ref,
                omega_b0_B - omega_b0_B_ref,
            ]
        )

        u_lqr = control_ref - K @ x_err

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
