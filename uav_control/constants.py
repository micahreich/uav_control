import numpy as np
from spatialmath.base import qeye

# Helpers used to index into the state vector
_R_B0_N_SIZE = 3
_Q_NB_SIZE = 4
_V_B0_N_SIZE = 3
_OMEGA_B0_B_SIZE = 3
_state_sizes_cumsum = [i for i in range(0, _R_B0_N_SIZE + _Q_NB_SIZE + _V_B0_N_SIZE + _OMEGA_B0_B_SIZE)]

R_B0_N = _state_sizes_cumsum[0:_R_B0_N_SIZE]
Q_NB = _state_sizes_cumsum[R_B0_N[-1] + 1 : R_B0_N[-1] + 1 + _Q_NB_SIZE]
V_B0_N = _state_sizes_cumsum[Q_NB[-1] + 1 : Q_NB[-1] + 1 + _V_B0_N_SIZE]
OMEGA_B0_B = _state_sizes_cumsum[V_B0_N[-1] + 1 : V_B0_N[-1] + 1 + _OMEGA_B0_B_SIZE]

def decompose_state(y: np.ndarray):
    """
    Decomposes the state vector into individual sub-arrays.

    Args:
        y (np.ndarray): The state vector.

    Returns:
        tuple: A tuple containing the following sub-arrays:
            - r_b0_N: The position vector of the center of mass, expressed in the world frame.
            - q_NB: The quaternion representing the body to world rotation.
            - v_b0_N: The linear velocity vector of the center of mass, expressed in the world frame.
            - omega_b0_B: The angular velocity vector of the body frame relative to the world frame, expressed in the body frame.
    """
    r_b0_N = y[R_B0_N]
    q_NB = y[Q_NB]
    v_b0_N = y[V_B0_N]
    omega_b0_B = y[OMEGA_B0_B]
    
    return r_b0_N, q_NB, v_b0_N, omega_b0_B

def decompose_state_dot(ydot: np.ndarray):
    """
    Decomposes the state derivative vector into individual components.

    Args:
        ydot (np.ndarray): The state derivative vector.

    Returns:
        tuple: A tuple containing the individual components of the state derivative vector:
            - v_b0_N: The velocity of the body frame origin, expressed in the world frame.
            - q_NB_dot: The derivative of the quaternion representing the body to world rotation.
            - a_b0_N: The acceleration of the body frame origin, expressed in the world frame.
            - omega_b0_B_dot: The derivative of the angular velocity of the body frame w.r.t the world frame, expressed in the body frame.
    """
    v_b0_N = ydot[R_B0_N]
    q_NB_dot = ydot[Q_NB]
    a_b0_N = ydot[V_B0_N]
    omega_b0_B_dot = ydot[OMEGA_B0_B]
    
    return v_b0_N, q_NB_dot, a_b0_N, omega_b0_B_dot

def compose_state(r_b0_N: np.ndarray = np.zeros(3),
                  q_NB: np.ndarray = qeye(),
                  v_b0_N: np.ndarray = np.zeros(3),
                  omega_b0_B: np.ndarray = np.zeros(3)):
    """
    Composes the state vector for a system.

    Args:
        r_b0_N (array_like): The position vector of the center of mass, expressed in the world frame. Shape (3,).
        q_NB (array_like): The quaternion representing the body to world rotation. Shape (4,).
        v_b0_N (array_like): The linear velocity vector of the center of mass, expressed in the world frame. Shape (3,).
        omega_b0_B (array_like): The angular velocity vector of the body frame relative to the world frame, expressed in the body frame. Shape (3,).

    Returns:
        ndarray: State vector containing the position, orientation, velocity, and angular velocity. Shape (13,).

    Notes:
        The state vector y is composed as follows:
        - y[0:3] contains the position vector r_b0_N.
        - y[3:7] contains the quaternion q_NB.
        - y[7:10] contains the velocity vector v_b0_N.
        - y[10:13] contains the angular velocity vector omega_b0_B.
    """
    y = np.empty(_state_sizes_cumsum[-1] + 1)
    
    y[R_B0_N] = r_b0_N
    y[Q_NB] = q_NB
    y[V_B0_N] = v_b0_N
    y[OMEGA_B0_B] = omega_b0_B
    
    return y

def compose_state_dot(v_b0_N: np.ndarray = np.zeros(3),
                      q_NB_dot: np.ndarray = np.zeros(3),
                      a_b0_N: np.ndarray = np.zeros(3),
                      omega_b0_B_dot: np.ndarray = np.zeros(3)):
    """
    Composes the state vector derivative for a system.

    Args:
        v_b0_N (array_like): The linear velocity vector of the center of mass, expressed in the world frame. Shape (3,).
        q_NB_dot (array_like): The derivative of the quaternion representing the body to world rotation. Shape (4,).
        a_b0_N (array_like): The linear acceleration vector of the center of mass, expressed in the world frame. Shape (3,).
        omega_b0_B_dot (array_like): The derivative of the angular velocity vector of the body frame relative to the world frame, expressed in the body frame. Shape (3,).

    Returns:
        ndarray: State vector derivative containing the velocity, orientation derivative, acceleration, and angular acceleration. Shape (13,).

    Notes:
        The state vector derivative ydot is composed as follows:
        - ydot[0:3] contains the velocity vector v_b0_N.
        - ydot[3:7] contains the derivative of the quaternion q_NB.
        - ydot[7:10] contains the acceleration vector a_b0_N.
        - ydot[10:13] contains the derivative of the angular velocity vector omega_b0_B.
    """
    ydot = np.empty(_state_sizes_cumsum[-1] + 1)
    
    ydot[R_B0_N] = v_b0_N
    ydot[Q_NB] = q_NB_dot
    ydot[V_B0_N] = a_b0_N
    ydot[OMEGA_B0_B] = omega_b0_B_dot
    
    return ydot

# Body axes
e3_B = np.array([0, 0, 1], dtype=np.float64)        
thrust_axis_B = e3_B

# Gravity
g = -9.8067 # m/s^2 downwards
a_g_N = np.array([0, 0, g])

# Global axes
e1_N = np.array([1, 0, 0], dtype=np.float64)
e2_N = np.array([0, 1, 0], dtype=np.float64)
e3_N = np.array([0, 0, 1], dtype=np.float64)