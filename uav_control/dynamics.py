from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d
import spatialmath as sm
from spatialmath.base import qnorm, qconj, qdotb, qvmul, qslerp, qunit, q2r, skewa

from hybrid_ode_sim.simulation.base import ContinuousTimeModel, DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from uav_control.constants import decompose_state, compose_state, compose_state_dot, g, a_g_N, e3_B, thrust_axis_B
from dataclasses import dataclass


@dataclass
class QuadrotorRigidBodyParams:
    m: float                               # mass
    I: np.ndarray                          # inertia matrix
    D_drag: np.ndarray                     # diagonal drag coefficient matrix
        

class QuadrotorRigidBodyDynamics(ContinuousTimeModel):
    def __init__(self, y0: np.ndarray, params: QuadrotorRigidBodyParams,
                 logging_level=LogLevel.ERROR):
        """
        Initializes the QuadrotorDynamics class.

        Args:
            y0 (np.ndarray): The initial state of the quadrotor. This should be a 13x1 vector with the following elements:
                - r_b0_N: 3x1 vector, the position of the quadrotor in the ENU frame
                - q_NB: 4x1 vector, the attitude of the quadrotor as a (unit) quaternion
                - v_b0_N: 3x1 vector, the velocity of the quadrotor in the ENU frame
                - omega_b0_B: 3x1 vector, the angular velocity of the quadrotor in the body frame
            params (QuadrotorRigidBodyParams): The parameters of the UAV rigid body model.
            logging_level (LogLevel, optional): The logging level for the model. Defaults to LogLevel.ERROR.
        """
        super().__init__(y0, 'quadrotor_state', params, logging_level=logging_level)
        self.I_inv = np.linalg.inv(params.I)
    
    def output_validate(self, y: np.ndarray) -> np.ndarray:
        """
        Validates the output of the quadrotor dynamics, ensuring unit-norm quaternions.

        Args:
            y (np.ndarray): The output to be validated.

        Returns:
            np.ndarray: The validated output.
        """

        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(y)
        
        return compose_state(
            r_b0_N,
            q_NB / qnorm(q_NB),
            v_b0_N,
            omega_b0_B
        )
        
    def continuous_dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculates the continuous dynamics of the quadrotor.

        Args:
            t (float): The current time.
            y (np.ndarray): The state vector of the quadrotor.

        Returns:
            np.ndarray: The derivative of the state vector.

        Notes:
            Computes the rigid body dynamics (state derivatives) of a quadrotor UAV based on a first-order drag model.
        """
        try:
            allocated_wrench = self.input_models['allocator'].y
        except KeyError:
            self.logger.warning("No input model found for 'allocator'. Defaulting to zero thrust and torque.")
            allocated_wrench = np.zeros(4)
        
        decomposed_state = decompose_state(y)
        allocated_collective_thrust = allocated_wrench[0]
        allocated_torque_B = allocated_wrench[1:]
        
        ydot = compose_state_dot(
            v_b0_N=self.compute_v_b0_N(decomposed_state),
            q_NB_dot=self.compute_q_NB_dot(decomposed_state),
            a_b0_N=self.compute_a_b0_N(decomposed_state, allocated_collective_thrust),
            omega_b0_B_dot=self.compute_omega_b0_B_dot(decomposed_state, allocated_torque_B)
        )
                
        return ydot
    
    def history(self) -> Tuple[np.ndarray, np.ndarray, interp1d]:
        """
        Returns the history of the quadrotor's state over time.

        Returns:
            Tuple[np.ndarray, np.ndarray, interp1d]: 
                - Array of time points
                - Array of state vectors
                - Interpolation function for the state
        """
        ts = np.array(self.t_history)
        ys = np.array(self.y_history)
        
        def interpolator_fn(t):
            assert t >= ts[0] and t <= ts[-1], f"Time {t} is outside the range of the simulation history."
            
            t_idx = np.searchsorted(ts, t)
            i_prev = max(0, t_idx - 1)
            i_next = min(len(ts) - 1, t_idx)
            
            if i_prev == i_next:
                return ys[i_prev]
            
            scale_prev = (ts[i_next] - t) / (ts[i_next] - ts[i_prev])
            scale_next = 1.0 - scale_prev
            
            r_b0_N_prev, q_NB_prev, v_b0_N_prev, omega_b0_B_prev = decompose_state(ys[i_prev])
            r_b0_N_next, q_NB_next, v_b0_N_next, omega_b0_B_next = decompose_state(ys[i_next])
            
            return compose_state(
                r_b0_N=r_b0_N_prev * scale_prev + r_b0_N_next * scale_next,
                q_NB=qunit(qslerp(q_NB_prev, q_NB_next, s=scale_next, shortest=True)),
                v_b0_N=v_b0_N_prev * scale_prev + v_b0_N_next * scale_next,
                omega_b0_B=omega_b0_B_prev * scale_prev + omega_b0_B_next * scale_next
            )
            
        return ts, ys, interpolator_fn

    def compute_v_b0_N(self, decomposed_state):
        """
        Computes the velocity of the quadrotor in the ENU frame.

        Args:
            decomposed_state (tuple): The decomposed state of the quadrotor.

        Returns:
            np.ndarray: The velocity of the quadrotor in the ENU frame.
        """
        _r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state
        
        return v_b0_N
    
    def compute_q_NB_dot(self, decomposed_state):
        """
        Computes the derivative of the quaternion representing the quadrotor's orientation.

        Args:
            decomposed_state (tuple): The decomposed state of the quadrotor.

        Returns:
            np.ndarray: The derivative of the quaternion.
        """
        _r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state
        
        # Quaternion derivative with angular velocity expressed in the body frame
        return qdotb(q_NB, omega_b0_B)
        
    def compute_a_b0_N(self, decomposed_state, allocated_collective_thrust):
        """
        Computes the acceleration of the quadrotor in the ENU frame.

        Args:
            decomposed_state (tuple): The decomposed state of the quadrotor.
            allocated_collective_thrust (float): The allocated collective thrust.

        Returns:
            np.ndarray: The acceleration of the quadrotor in the ENU frame.
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

        a_b0_N = 1/self.params.m * (f_g_N + f_control_N + f_drag_N)
        
        return a_b0_N

    def compute_j_b0_N(self, decomposed_state, allocated_collective_thrust, allocated_collective_thrust_dot):
        """
        Computes the jerk of the quadrotor in the ENU frame.

        Args:
            decomposed_state (tuple): The decomposed state of the quadrotor.
            allocated_collective_thrust (float): The allocated collective thrust.
            allocated_collective_thrust_dot (float): The derivative of the allocated collective thrust.

        Returns:
            np.ndarray: The jerk of the quadrotor in the ENU frame.
        """
        r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state
        a_b0_N = self.compute_a_b0_N(decomposed_state, allocated_collective_thrust)
        
        rot_NB = q2r(q_NB)
        rot_NB_dot = rot_NB @ skewa(omega_b0_B)
        
        _, _, b3 = rot_NB.T
        _, _, b3_dot = rot_NB_dot.T
        
        f_control_N_dot = allocated_collective_thrust_dot * b3 + \
                          allocated_collective_thrust * b3_dot
        
        f_drag_N_dot = (rot_NB_dot @ self.params.D_drag @ rot_NB.T - \
                        rot_NB @ self.params.D_drag @ rot_NB.T @ rot_NB_dot @ rot_NB.T) @ -v_b0_N + \
                       (rot_NB @ self.params.D_drag @ rot_NB.T) @ -a_b0_N
        
        j_b0_N = 1/self.params.m * (f_control_N_dot + f_drag_N_dot)
        
        return j_b0_N
        
    def compute_omega_b0_B_dot(self, decomposed_state, allocated_torque_B):
        """
        Computes the derivative of the angular velocity of the quadrotor in the body frame.

        Args:
            decomposed_state (tuple): The decomposed state of the quadrotor.
            allocated_torque_B (np.ndarray): The allocated torque in the body frame.

        Returns:
            np.ndarray: The derivative of the angular velocity in the body frame.
        """
        _r_b0_N, q_NB, v_b0_N, omega_b0_B = decomposed_state
        
        # Euler's rotation equation for rigid bodies
        domega_b0_B = self.I_inv @ (allocated_torque_B - np.cross(omega_b0_B, self.params.I @ omega_b0_B))
        
        return domega_b0_B