import numpy as np
from spatialmath.base import qeye

# Helpers used to index into the state vector
R_B0_N_DIM = 3
Q_NB_DIM = 4
V_B0_N_DIM = 3
OMEGA_B0_B_DIM = 3
_state_sizes_cumsum = [i for i in range(0, R_B0_N_DIM + Q_NB_DIM + V_B0_N_DIM + OMEGA_B0_B_DIM)]

# Helpers used to index into the control vector
THRUST_DIM = 1
TAU_B0_B_DIM = 3
_control_sizes_cumsum = [i for i in range(0, THRUST_DIM + TAU_B0_B_DIM)]

R_B0_N = _state_sizes_cumsum[0:R_B0_N_DIM]
Q_NB = _state_sizes_cumsum[R_B0_N[-1] + 1 : R_B0_N[-1] + 1 + Q_NB_DIM]
V_B0_N = _state_sizes_cumsum[Q_NB[-1] + 1 : Q_NB[-1] + 1 + V_B0_N_DIM]
OMEGA_B0_B = _state_sizes_cumsum[V_B0_N[-1] + 1 : V_B0_N[-1] + 1 + OMEGA_B0_B_DIM]

THRUST = _control_sizes_cumsum[0:THRUST_DIM]
TAU_B0_B = _control_sizes_cumsum[THRUST[-1] + 1 : THRUST[-1] + 1 + TAU_B0_B_DIM]


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


def decompose_control(u: np.ndarray):
    """
    Decomposes the control vector into individual sub-arrays

    Parameters
    ----------
    u : np.ndarray
        Stacked control vector

    Returns
    -------
    thrust : float
        The collective thrust from the rotors (Force)
    tau_b0_B: : np.ndarray
        The torque exterted on the body-frame expressed from an observer in the body frame
    """
    thrust = u[THRUST][0]
    tau_b0_B = u[TAU_B0_B]

    return thrust, tau_b0_B


def compose_state(
    r_b0_N: np.ndarray = np.zeros(3),
    q_NB: np.ndarray = qeye(),
    v_b0_N: np.ndarray = np.zeros(3),
    omega_b0_B: np.ndarray = np.zeros(3),
):
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


def compose_state_dot(
    v_b0_N: np.ndarray = np.zeros(3),
    q_NB_dot: np.ndarray = np.zeros(3),
    a_b0_N: np.ndarray = np.zeros(3),
    omega_b0_B_dot: np.ndarray = np.zeros(3),
):
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


def compose_control(c: float = 0.0, tau_b0_B: np.ndarray = np.zeros(3)):
    """
    Composes the stacked control vector for a quadrotor

    Parameters
    ----------
    c : float, optional
        Collective thrust, by default 0.0
    tau_b0_B : np.ndarray, optional
        Torque exterted on the body as observed by an observer in the body-frame, by default np.zeros(3)

    Returns
    -------
    u : np.ndarray
        The stacked control vector as a np array
    """
    u = np.empty(_control_sizes_cumsum[-1] + 1)
    u[THRUST] = c
    u[TAU_B0_B] = tau_b0_B

    return u


# Body axes
e1 = np.array([1, 0, 0], dtype=np.float64)
e2 = np.array([0, 1, 0], dtype=np.float64)
e3 = np.array([0, 0, 1], dtype=np.float64)

thrust_axis_B = e3

# Gravity
g = -9.8067  # m/s^2 downwards
a_g_N = np.array([0, 0, g])
