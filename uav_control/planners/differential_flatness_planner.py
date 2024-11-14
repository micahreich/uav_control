from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import spatialmath as sm
from hybrid_ode_sim.simulation.base import DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from spatialmath.base import q2r, qconj, qdotb, qnorm, qvmul, r2q, rotx, roty, rotz, skewa

from uav_control.constants import R_B0_N, V_B0_N, a_g_N, compose_control, compose_state, e1
from uav_control.controllers.geometric_controller import decompose_state


class DifferentialFlatnessPlanner(DiscreteTimeModel):
    def __init__(self, y0: Any, sample_rate: int, planner_name: str):
        super().__init__(y0, sample_rate, "dfb_planner", None, logging_level=LogLevel.INFO)
        self.planner_name = planner_name

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        """
        Returns the desired position, velocity, orientation, collective thrust magnitude, and angular velocity
        based on differential flatness of the dynamics model.

        Parameters
        ----------
        t : float
            Current simulation time
        _y : Any
            N/A

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[float, np.ndarray]
            Desired position, velocity, orientation (unit quaternion) in world frame.
            Desired collective thrust magnitude and angular velocity in body frame.
        """
        dynamics = self.input_models["quadrotor_state"]
        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(dynamics.y)
        rot_NB = q2r(q_NB)

        # The planner must output differential flatness states
        [r_b0_N_ref, v_b0_N_ref, a_b0_N_ref, j_b0_N_ref, s_b0_N_ref], [
            b_1d,
            b_1d_dot,
            b_1d_ddot,
        ] = self.input_models[self.planner_name].y

        # Compute desired state (position, velocity, orientation) and control (thrust, angular velocity)
        # using differential flatness properties of quadrotor UAVs

        A = a_b0_N_ref

        [r_d0_N, v_d0_N, rot_ND, omega_d0_D], [
            f_cmd,
            tau_d0_D,
        ] = dynamics.compute_diferential_flatness_states_controls(
            r_b0_N_ref,
            v_b0_N_ref,
            A,
            j_b0_N_ref,
            s_b0_N_ref,
            b_1d,
            b_1d_dot,
            b_1d_ddot,
        )

        rot_BD = rot_NB.T @ rot_ND

        # Desired thrust along body frame b3 axis
        b_3 = rot_NB[:, 2]
        thrust_d = np.dot(f_cmd, b_3)

        # Desired angular velocity (in body frame)
        omega_d0_B = rot_BD @ omega_d0_D

        # Desired torque (in body frame)
        tau_d0_B = rot_BD @ tau_d0_D

        desired_state = compose_state(
            r_b0_N=r_d0_N,
            q_NB=r2q(rot_ND),
            v_b0_N=v_d0_N,
            omega_b0_B=omega_d0_B,
        )

        desired_control = compose_control(
            c=thrust_d,
            tau_b0_B=tau_d0_B,
        )

        return desired_state, desired_control
