from dataclasses import dataclass, field
from typing import Any, Dict, List

import cvxpy as cp
import numpy as np
from hybrid_ode_sim.simulation.base import DiscreteTimeModel

from uav_control.constants import (
    a_g_N,
    compose_state_dot,
    decompose_state,
    e3_B,
    e3_N,
    g,
    thrust_axis_B,
)
from uav_control.dynamics import QuadrotorRigidBodyParams


@dataclass
class QuadrotorQPAllocatorParams:
    W: np.ndarray = field(
        default_factory=lambda: np.eye(4)
    )  # 4x4 diagonal weight matrix
    l: float = 0.2  # arm length
    u_min: float = 0.0  # minimum rotor thrust
    u_max: float = np.inf  # maximum rotor thrust
    cq_ct_ratio: float = 1e-2  # ratio of thrust to torque coefficient


class QuadrotorQPAllocator(DiscreteTimeModel):
    def __init__(
        self, y0: np.ndarray, sample_rate: int, params=QuadrotorQPAllocatorParams()
    ):
        """
        Initialize the QP-based control allocator. Solves the following optimization problem:

        Given the control wrench tau_des, find the control u (motor thrusts) within bounds to minimize the cost function
        tau_err^T @ W @ tau_err, where tau_err = tau_des - G1 @ u and W is a diagonal weight matrix.

        Args:
            y0 (np.ndarray): The initial state of the allocator, should be a 4x1 vector with the following elements:
                - collective_thrust: float, the collective thrust of the quadrotor
                - torques: 3x1 vector, the torques to be applied to the quadrotor
            sample_rate (int): The sample rate of the control system (Hz).
            params (QuadrotorQPAllocatorParams, optional): The parameters for the QPControlAllocator, by default QuadrotorQPAllocatorParams().
        """
        super().__init__(y0, sample_rate, "allocator", params)

        # Construct the allocation matrix G s.t. tau = G @ u
        self.G1 = np.zeros(shape=(4, 4))
        self.G1[0, :] = 1.0

        for i, r in enumerate(
            [
                self.params.l * np.array([1.0, 0.0, 0.0]),
                self.params.l * np.array([0.0, 1.0, 0.0]),
                self.params.l * np.array([-1.0, 0.0, 0.0]),
                self.params.l * np.array([0.0, -1.0, 0.0]),
            ]
        ):
            self.G1[1:, i] = np.cross(r, e3_B)

        self.G1[3, :] += [
            self.params.cq_ct_ratio,
            -self.params.cq_ct_ratio,
            self.params.cq_ct_ratio,
            -self.params.cq_ct_ratio,
        ]

    def discrete_dynamics(self, _t: float, _y: np.ndarray) -> np.ndarray:
        """
        Calculate the allocated control output within the control bounds.

        Args:
            _t (float): Current time (not used in this function).
            _y (np.ndarray): The current state of the allocator (not used in this function).

        Returns:
            np.ndarray: The allocated control output of the allocator; a 4x1 vector with the following elements:
                - collective_thrust: float, the collective thrust of the quadrotor
                - torques: 3x1 vector, the torques to be applied to the quadrotor

        Raises:
            ValueError: If the QP allocation problem is infeasible or unbounded.
        """
        control_wrench = self.input_models["controller"].y

        if self.params.u_min == -np.inf and self.params.u_max == np.inf:
            return control_wrench

        # Build the optimization problem, where u is the control input (motor thrusts/forces)
        u = cp.Variable(4)

        constraint_matrix = np.vstack((np.eye(4), -np.eye(4)))

        constraint_vector = np.hstack(
            (self.params.u_max * np.ones(4), -self.params.u_min * np.ones(4))
        )

        P = self.G1.T @ self.params.W @ self.G1
        q = -2 * self.G1.T @ self.params.W @ control_wrench

        prob = cp.Problem(
            cp.Minimize(cp.quad_form(u, P) + q.T @ u),
            [constraint_matrix @ u <= constraint_vector],
        )
        prob.solve(solver="SCS")

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError(f"QP allocation had status {prob.status}!")

        # Return the allocated collective thrust and body torque
        return self.G1 @ u.value
