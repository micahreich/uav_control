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

import jax
import jax.numpy as jnp
import jaxlie as jlie

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
    def to_lqr_state(x, u):
        r_b0_W, q_WB, v_b0_W, omega_b0_B = decompose_state(x)
        c_B, _ = decompose_control(u)

        return np.concatenate([r_b0_W, q_WB, v_b0_W]), np.concatenate([[c_B], omega_b0_B])

    @staticmethod
    def compute_lqr_error_state(lqr_x, lqr_x0, lqr_u, lqr_u0) -> Tuple[np.ndarray, np.ndarray]:
        r_d0_W, q_WD, v_d0_W = LQRDynamicsLinearization.decompose_lqr_state(lqr_x0)
        r_b0_W, q_WB, v_b0_W = LQRDynamicsLinearization.decompose_lqr_state(lqr_x)

        c_D, omega_d0_B = LQRDynamicsLinearization.decompose_lqr_control(lqr_u0)
        c_B, omega_b0_B = LQRDynamicsLinearization.decompose_lqr_control(lqr_u)

        delta_p = r_b0_W - r_d0_W
        delta_R = q2r(qqmul(qconj(q_WD), q_WB))
        delta_theta = r2x(delta_R, representation="exp")
        delta_v = v_b0_W - v_d0_W

        delta_omega = omega_b0_B - omega_d0_B
        delta_c = c_B - c_D

        delta_lqr_x = np.concatenate([delta_p, delta_theta, delta_v])
        delta_lqr_u = np.concatenate([delta_c, delta_omega])

        return delta_lqr_x, delta_lqr_u

    @staticmethod
    def decompose_lqr_error_state(delta_lqr_x, delta_lqr_u):
        delta_p, delta_theta, delta_v = delta_lqr_x[:3], delta_lqr_x[3:6], delta_lqr_x[6:9]
        delta_c, delta_omega = delta_lqr_u[:1], delta_lqr_u[1:]

        return [delta_p, delta_theta, delta_v], [delta_c, delta_omega]

    @staticmethod
    def decompose_lqr_state(lqr_x):
        p, q, v = lqr_x[:3], lqr_x[3:7], lqr_x[7:]
        return [p, q, v]

    @staticmethod
    def decompose_lqr_control(lqr_u):
        c, omega = lqr_u[:1], lqr_u[1:]
        return [c, omega]

    def _linearize_error_state_dynamics(self, x, x0, u, u0):
        lqr_x0, lqr_u0 = self.to_lqr_state(x0, u0)
        lqr_x, lqr_u = self.to_lqr_state(x, u)

        [_, q0, _], [c0, _] = self.decompose_lqr_state(lqr_x0), self.decompose_lqr_control(lqr_u0)
        [delta_p, delta_theta, delta_v], [delta_c, delta_omega] = self.decompose_lqr_error_state(
            *self.compute_lqr_error_state(lqr_x, lqr_x0, lqr_u, lqr_u0)
        )

        m = self.input_models["quadrotor_state"].params.m

        def jax_skew(v):
            return jnp.array([[0, -v[2], v[1]],
                              [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]])

        def error_state_dynamics(delta_p, delta_theta, delta_v, delta_c, delta_omega):
            q0_j = jlie.SO3(q0)
            delta_q_j = jlie.SO3.exp(delta_theta)
            q_j = q0_j.multiply(delta_q_j)

            c_j = jnp.array([0., 0., c0[0] + delta_c[0]])
            c0_j = jnp.array([0., 0., c0[0]])

            delta_theta_skew_j = jax_skew(delta_theta)
            delta_theta_norm = jnp.linalg.norm(delta_theta)

            J_r_inv = jnp.eye(3) + \
                    1/2 * delta_theta_skew_j + \
                    (1/delta_theta_norm**2 - (1 + jnp.cos(delta_theta_norm))/(2 * delta_theta_norm * jnp.sin(delta_theta_norm))) * delta_theta_skew_j @ delta_theta_skew_j

            delta_p_dot = delta_v
            delta_theta_dot = J_r_inv @ delta_omega
            delta_v_dot = 1/m * (q_j.apply(c_j) - \
                                 q_j.apply(c0_j))

            return jnp.concatenate([delta_p_dot, delta_theta_dot, delta_v_dot])

        # Compute the Jacobians of the error state dynamics wrt error-state, error-control

        A = np.hstack(
            jax.jacfwd(error_state_dynamics, argnums=[0, 1, 2])(
                delta_p, delta_theta, delta_v, delta_c, delta_omega
            )
        )

        B = np.hstack(
            jax.jacfwd(error_state_dynamics, argnums=[3, 4])(
                delta_p, delta_theta, delta_v, delta_c, delta_omega
            )
        )

        return A, B

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        curr_state = self.input_models["quadrotor_state"].y
        curr_control = self.input_models["controller"].y

        desired_state, desired_control = self.input_models[self.planner_name].y

        A, B = self._linearize_error_state_dynamics(curr_state, desired_state, curr_control, desired_control)

        # Add some damping to the system to ensure stability (Re(eig(A)) < 0)
        damping = 0.01
        A_stable = A - damping * np.eye(A.shape[0])

        P = sp.linalg.solve_continuous_are(A_stable, B, self.params.Q, self.params.R)
        K = self.params.R_inv @ B.T @ P

        return LQRGainSolverState(
            Jx=A,
            Ju=B,
            x0=desired_state,
            u0=desired_control,
            K=K,
        )


class LQRController(DiscreteTimeModel):
    def __init__(
        self,
        sample_rate: int,
        y0=np.zeros(nu),
    ):
        super().__init__(y0, sample_rate, "lqr_controller")

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        curr_state = self.input_models["quadrotor_state"].y
        lqr_gains: LQRGainSolverState = self.input_models["lqr_linearization"].y

        lqr_x0, lqr_u0 = LQRDynamicsLinearization.to_lqr_state(lqr_gains.x0, lqr_gains.u0)
        lqr_x, lqr_u = LQRDynamicsLinearization.to_lqr_state(curr_state, y)

        delta_lqr_x, _ = LQRDynamicsLinearization.compute_lqr_error_state(lqr_x, lqr_x0, lqr_u, lqr_u0)

        # delta_lqr_x, _, _, lqr_u0 = LQRDynamicsLinearization.compute_reduced_error_state(curr_state,
        #                                                                                  lqr_gains.x0,
        #                                                                                  y,
        #                                                                                  lqr_gains.u0)

        # r_d0_W, q_WD, v_d0_W, omega_d0_B = decompose_state(lqr_gains.x0)
        # c_D, _ = decompose_control(lqr_gains.u0)
        # lqr_u0 = np.concatenate([[c_D], omega_d0_B])

        # Compute the LQR control law, u = -K * delta_x + u0
        # where u0 is the control input (which, for the LQR system is the collective
        # thrust and body-frame angular velocity) at the linearization point
        delta_lqr_u = -lqr_gains.K @ delta_lqr_x
        lqr_u = delta_lqr_u + lqr_u0

        return lqr_u

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
        control_tau = dynamics.params.I @ omega_c0_B_dot + np.cross(
            omega_b0_B, dynamics.params.I @ omega_b0_B
        )

        return np.concatenate([[collective_thrust_c], control_tau])
