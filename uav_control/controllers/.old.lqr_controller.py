from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import control
import numpy as np
import scipy as sp
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from spatialmath.base import (angvec2r, q2r, qconj, qnorm, qqmul, qvmul, r2q,
                              skewa)

from uav_control.constants import (a_g_N, compose_state_dot, decompose_state,
                                   e3, g, thrust_axis_B)
from uav_control.dynamics import QuadrotorRigidBodyParams

nx, nu = 3 + 3 + 4, 1 + 3


@dataclass
class LQRGainSolverParams:
    Q: np.ndarray = field(default_factory=lambda: np.eye(nx))  # 10x10 state cost matrix
    R: np.ndarray = field(default_factory=lambda: np.eye(nu))  # 4x4 control cost matrix


@dataclass
class LQRGainSolverState:
    Jx: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nx)))  # A matrix
    Ju: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nu)))  # B matrix
    x0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx,)))  # linearization state
    u0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nu,)))  # linearization control
    K: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nu, nx)))  # LQR gain matrix


@dataclass
class BodyrateControllerParams:
    P: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 proportional gain matrix


class LQRGainSolver(DiscreteTimeModel):
    def __init__(
        self,
        y0: LQRGainSolverState,
        sample_rate: int,
        params: LQRGainSolverParams,
        rbd_params: QuadrotorRigidBodyParams,
    ):
        # https://underactuated.mit.edu/lqr.html
        super().__init__(y0, sample_rate, "lqr_gains", params)
        self.rbd_params = rbd_params

        if np.allclose(np.linalg.det(self.params.R), 0):
            raise ValueError("LQR gain parameter R must be invertible")

        self.R_inv = np.linalg.inv(self.params.R)

    @staticmethod
    def lqr_state(state_ref: np.ndarray) -> np.ndarray:
        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(state_ref)
        return np.concatenate([r_b0_N, v_b0_N, q_NB])

    @staticmethod
    def dqu_dq(q: np.ndarray) -> np.ndarray:
        """
        dqu_dq Compute the derivative of the unit quaternion qu with respect to quaternion q

        Parameters
        ----------
        q : np.ndarray
            Quaternion (not necesarrily unit norm)

        Returns
        -------
        np.ndarray (4, 4)
            Jacobian of the unit quaternion with respect to quaternion
        """
        q_norm = qnorm(q)

        return 1 / q_norm * (np.eye(4) - 1 / (q_norm**2) * np.outer(q, q))

    def linearize_dynamics(
        self, linearization_state, linearization_control
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearizes the system dynamics around the given state and control input

        This method computes the Jacobian matrices representing the linearized system dynamics
        of the form \dot{x} = J_x * x + J_u * u, where J_x is the Jacobian with respect to the
        state, and J_u is the Jacobian with respect to the control input

        Parameters
        ----------
        linearization_state : np.ndarray
            The current state of the system, including position, orientation (quaternion),
            velocity
        linearization_control : np.ndarray
            The control input to the system, including collective thrust and angular velocities
            (omega_c0_B)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Jx : np.ndarray (nx, nx)
                The Jacobian matrix of the system with respect to the state variables
            Ju : np.ndarray (nx, nu)
                The Jacobian matrix of the system with respect to the control input
        """
        collective_thrust, omega_c0_B = linearization_control[0], linearization_control[1:4]
        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(linearization_state)
        qw, qx, qy, qz = q_NB
        wx, wy, wz = omega_c0_B

        # Construct jacobians J_x, J_u to represent the linearized system dynamics as \dot{x} = J_x * x + J_u * u
        Jx = np.zeros(shape=(nx, nx))
        Ju = np.zeros(shape=(nx, nu))

        dqu_dq = LQRGainSolver.dqu_dq(q_NB)
        assert dqu_dq.shape == (4, 4)

        # Compute entries for d\dot{p}/dv
        dpdot_dv = np.eye(3)

        assert dpdot_dv.shape == (3, 3)

        # Compute entries for d\dot{v}/dv TODO: when adding drag model, this will matter

        # Compute entries for d\dot{v}/dq
        dvdot_dqu = (
            2
            * 1
            / self.rbd_params.m
            * collective_thrust
            * np.array(
                [[qy, qz, qw, qx], [-qx, -qw, qz, qy], [0, -2 * qx, -2 * qy, 0]], dtype=np.float64
            )
        )
        dvdot_dq = dvdot_dqu @ dqu_dq

        assert dvdot_dq.shape == (3, 4)

        # Compute entries for d\dot{q}/dq
        dqdot_dqu = (
            1
            / 2
            * np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])
        )
        dqdot_dq = dqdot_dqu @ dqu_dq

        assert dqdot_dq.shape == (4, 4)

        # Compute entries for d\dot{v}/dc
        dvdot_dc = (
            2
            * 1
            / self.rbd_params.m
            * np.array([[qw * qy + qx * qz], [qy * qz - qw * qx], [1 / 2 - qx**2 - qy**2]])
        )

        assert dvdot_dc.shape == (3, 1)

        # Compute entries for d\dot{q}/domega_c
        dqdot_domega_c = (
            1 / 2 * np.array([[-qx, -qy, -qz], [qw, -qz, -qy], [qz, qw, qx], [-qy, qx, qw]])
        )

        assert dqdot_domega_c.shape == (4, 3)

        # Assemble Jacobian matrices Jx, Ju; \dot{x} = J_x * x + J_u * u
        # fmt: off
        Jx = np.block([
            # d?/dp             d?/dv             d?/dq
            [np.zeros((3, 3)), dpdot_dv,         np.zeros((3, 4))], # d\dot{p}/d?
            [np.zeros((3, 3)), np.zeros((3, 3)), dvdot_dq],         # d\dot{v}/d?
            [np.zeros((4, 3)), np.zeros((4, 3)), dqdot_dq]          # d\dot{q}/d?
        ])

        Ju = np.block([
            # d?/dc             d?/domega_c
            [np.zeros((3, 1)), np.zeros((3, 3))], # d\dot{p}/d?
            [dvdot_dc,         np.zeros((3, 3))], # d\dot{v}/d?
            [np.zeros((4, 1)), dqdot_domega_c]    # d\dot{q}/d?
        ])
        # fmt: on

        return (Jx, Ju)

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        # Want to drive the system xerr = x - x_ref to zero, where error dynamics are given by
        # x_err_dot = f(x, u) - f(x_ref, u_ref), approximated as x_err_dot = f(x0, u0) + Jx (x - x0) + Ju (u - u0) - f(x_ref, u_ref)
        # Since we are linearizing around the reference state, control, we can approximate the error dynamics as
        # x_err_dot = Jx (x - x0) + Ju (u - u0)
        state_ref, control_ref = self.input_models["dfb_planner"].y

        Jx, Ju = self.linearize_dynamics(
            linearization_state=state_ref, linearization_control=control_ref
        )

        P = sp.linalg.solve_continuous_are(Jx, Ju, self.params.Q, self.params.R)
        K = self.R_inv @ Ju.T @ P

        return LQRGainSolverState(Jx=Jx, Ju=Ju, x0=state_ref, u0=control_ref, K=K)


class LQRController(DiscreteTimeModel):
    def __init__(
        self,
        y0: np.ndarray,
        sample_rate: int,
        rbd_params: QuadrotorRigidBodyParams,
    ):
        super().__init__(y0, sample_rate, "lqr_controller", None)
        self.rbd_params = rbd_params

    def discrete_dynamics(self, _t: float, _y: np.ndarray) -> np.ndarray:
        # Desired state is [x, v, q], virtual input is omega
        lqr_gains: LQRGainSolverState = self.input_models["lqr_gains"].y
        curr_state = self.input_models["quadrotor_state"].y
        _r_b0_N, _q_NB, _v_b0_N, omega_b0_B = decompose_state(curr_state)

        # LQR controller output given by: u = u0 - K (x - x0)
        u_lqr = lqr_gains.u0 - lqr_gains.K @ (LQRGainSolver.lqr_state(curr_state) - lqr_gains.x0)

        # Low-level bodyrate controller to track desired angular velocities
        collective_thrust_c, omega_c0_B = u_lqr[0], u_lqr[1:4]
        control_wrench = self.rbd_params.I @ -self.params.kP @ (omega_b0_B - omega_c0_B) + np.cross(
            omega_b0_B, self.rbd_params.I @ omega_b0_B
        )

        return u_lqr


class BodyrateController(DiscreteTimeModel):
    def __init__(
        self,
        y0: Any,
        sample_rate: int,
        params: BodyrateControllerParams,
        rbd_params: QuadrotorRigidBodyParams,
        logging_level=LogLevel.ERROR,
    ):
        super().__init__(y0, sample_rate, "bodyrate_controller", params, logging_level)
        self.rbd_params = rbd_params

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        lqr_control = self.input_models["lqr_controller"].y
        curr_state = self.input_models["quadrotor_state"].y

        _r_b0_N, _q_NB, _v_b0_N, omega_b0_B = decompose_state(curr_state)

        collective_thrust_c, omega_c0_B = lqr_control[0], lqr_control[1:4]
        omega_d0_B = -self.params.P @ (omega_b0_B - omega_c0_B)

        # Low-level bodyrate controller to track desired angular velocities, feedback linearization
        control_wrench = self.rbd_params.I @ omega_d0_B + np.cross(
            omega_b0_B, self.rbd_params.I @ omega_b0_B
        )
