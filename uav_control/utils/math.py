import numpy as np
import sympy as sym
from spatialmath.base import qnorm, skew, qconj, qqmul, qvmul

### Sympy functions for working with quaternions and vectors

sym_H = sym.BlockMatrix([[sym.zeros(1, 3)], [sym.eye(3)]]).as_explicit()


def sym_Lq(q):  # quaternion multiplication left matrix
    return sym.BlockMatrix(
        [
            [sym.Matrix([q[0]]), -q[1:, :].T],
            [q[1:, :], sym.eye(3) * q[0] + sym_skewsym(q[1:, :])],
        ]
    ).as_explicit()


def sym_Rq(q):  # quaternion multiplication right matrix
    return sym.BlockMatrix(
        [
            [sym.Matrix([q[0]]), -q[1:, :].T],
            [q[1:, :], sym.eye(3) * q[0] - sym_skewsym(q[1:, :])],
        ]
    ).as_explicit()


def sym_Aq(q):  # quaternion multiplication matrix (3x3 DCM)
    L = sym_Lq(q)
    R = sym_Rq(q)

    return sym_H.T @ L @ R.T @ sym_H


def sym_Gq(q):  # attitude jacobian for derivative of f: S^3 -> R^p
    return sym_Lq(q) @ sym_H


def sym_skewsym(v):  # skew symmetric matrix for 3-vector
    return sym.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


### Numpy functions for working with quaternions and vectors


def vee(M):
    """
    vee Return the 3-vector of a skew-symmetric matrix M

    Parameters
    ----------
    M : np.ndarray (3, 3)
        Skew-symmetric matrix

    Returns
    -------
    omega: np.ndarray (3,)
        3-vector of the skew-symmetric matrix
    """
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def skewsym(v):
    """
    skewsym Return skew symmetric matrix for 3-vector

    Parameters
    ----------
    v : np.ndarray (3,)
        3-vector

    Returns
    -------
    M : np.ndarray (3, 3)
        Skew symmetric matrix for v
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def compute_unit_vector_dot(u, u_dot):
    """
    compute_unit_vector_dot Compute the time derivative of a unit vector v, where
    v = u / ||u|| and u_dot is the time derivative of u

    Parameters
    ----------
    u : np.ndarray (n,)
        Un-normalized vector u
    u_dot : np.ndarray (n,)
        Time derivative of u

    Returns
    -------
    np.ndarray (n,)
        Time derivative of the unit vector v = u / ||u||
    """
    u_norm = np.linalg.norm(u)
    return u_dot / u_norm - np.dot(u, u_dot) / u_norm**3 * u


def compute_unit_vector_ddot(u, u_dot, u_ddot):
    """
    compute_unit_vector_ddot Compute the second time derivative of a unit vector v, where
    v = u / ||u|| and u_dot is the time derivative of u

    Parameters
    ----------
    u : np.ndarray (n,)
        Un-normalized vector u
    u_dot : np.ndarray (n,)
        Time derivative of u
    u_ddot : np.ndarray (n,)
        Second time derivative of u

    Returns
    -------
    np.ndarray (n,)
        Second time derivative of the unit vector v = u / ||u||
    """

    u_norm = np.linalg.norm(u)
    u_dot_norm = np.dot(u, u_dot) / u_norm
    return (
        (1 / u_norm) * u_ddot
        - (2 * np.dot(u, u_dot) / u_norm**3) * u_dot
        - ((u_dot_norm**2 + np.dot(u, u_ddot)) / u_norm**3) * u
        + (3 * np.dot(u, u_dot) ** 2 / u_norm**5) * u
    )


def compute_cross_product_dot(u, u_dot, v, v_dot):
    """
    compute_cross_product_dot Compute the time derivative of the cross product of two vectors u and v,
    u x v, given their time derivatives u_dot and v_dot

    Parameters
    ----------
    u : np.ndarray (3,)
        First vector
    u_dot : np.ndarray (3,)
        Time derivative of the first vector
    v : np.ndarray (3,)
        Second vector
    v_dot : np.ndarray (3,)
        Time derivative of the second

    Returns
    -------
    np.ndarray (3,)
        Time derivative of the cross product of u and v, u x v
    """
    return np.cross(u_dot, v) + np.cross(u, v_dot)


def dxu_dx_jacobian(x: np.ndarray) -> np.ndarray:
    """
    dqu_dq Compute the derivative of the x / ||x|| with respect to the vector x

    Parameters
    ----------
    x : np.ndarray
        Input vector

    Returns
    -------
    np.ndarray (4, 4)
        Jacobian of the unit vector with respect to vector x
    """
    d = x.size
    x_norm = np.linalg.norm(x)

    return (np.eye(d) * x_norm - np.outer(x, x)) / x_norm**2


def dxnorm_dx_jacobian(x: np.ndarray) -> np.ndarray:
    """
    dxnorm_dx_jacobian Compute the derivative of the norm of a vector x with respect to x

    Parameters
    ----------
    x : np.ndarray
        Input vector

    Returns
    -------
    np.ndarray (n, n)
        Jacobian of the norm with respect to x
    """
    x_norm = np.linalg.norm(x)
    return x / x_norm


def qu_to_rodgigues_params(q: np.ndarray) -> np.ndarray:
    """
    qu_to_rodgigues_params Convert a unit quaternion to Rodrigues parameters

    Parameters
    ----------
    q : np.ndarray
        Unit quaternion

    Returns
    -------
    np.ndarray (3,)
        Rodrigues parameters
    """
    return q[1:] / q[0]


def sign(x):
    """
    Returns the sign of a number.

    Parameters
    ----------
    x : float or int
        The number to determine the sign of.

    Returns
    -------
    float
        1.0 if x is positive or zero, -1.0 if x is negative.
    """
    if x >= 0:
        return 1.0
    else:
        return -1.0


def qu_to_aa(q):
    """
    Convert a unit quaternion to an axis-angle representation.

    Parameters
    ----------
    qu : np.ndarray
        Unit quaternion

    Returns
    -------
    np.ndarray, float
        Axis-angle representation as (axis, angle) pair
    """
    qu = q / qnorm(q)

    if qu[0] < 0:
        qu = -qu

    theta = 2 * np.arccos(qu[0])
    if np.abs(theta) < 1e-8:
        return np.random.normal(0, 1e-4, 3), 0.0

    axis = qu[1:] / np.sin(theta / 2)
    return axis, theta


def aa_to_dcm(axis, angle):
    """
    Convert an axis-angle representation to a direction cosine matrix.

    Parameters
    ----------
    axis : np.ndarray
        Axis of rotation
    angle : float
        Angle of rotation

    Returns
    -------
    np.ndarray
        Direction cosine matrix
    """
    axis_skew = skew(axis)
    return np.eye(3) + np.sin(angle) * axis_skew + (1 - np.cos(angle)) * (axis_skew @ axis_skew)


def so3_jacobian_left(axis, angle):
    """
    Compute the left Jacobian of the SO(3) group.

    Parameters
    ----------
    axis : np.ndarray
        Axis of rotation
    angle : float
        Angle of rotation

    Returns
    -------
    np.ndarray
        Left Jacobian of the SO(3) group
    """
    axis_skew = skew(axis)
    return (
        np.eye(3)
        + (1 - np.cos(angle)) / angle**2 * axis_skew
        + (angle - np.sin(angle)) / angle**3 * (axis_skew @ axis_skew)
    )


def so3_jacobian_right(axis, angle):
    """
    Compute the right Jacobian of the SO(3) group.

    Parameters
    ----------
    axis : np.ndarray
        Axis of rotation
    angle : float
        Angle of rotation

    Returns
    -------
    np.ndarray
        Right Jacobian of the SO(3) group
    """
    axis_skew = skew(axis)
    return (
        np.eye(3)
        - (1 - np.cos(angle)) / angle**2 * axis_skew
        + (angle - np.sin(angle)) / angle**3 * (axis_skew @ axis_skew)
    )

def qu_err_to_aa_err(q_curr, q_des, curr_minus_ref: bool = True):
    """
    Convert a quaternion error to an axis-angle error.

    Parameters
    ----------
    q_err : np.ndarray
        Quaternion error

    Returns
    -------
    np.ndarray
        Axis-angle error
    """
    q_err = qqmul(qconj(q_curr), q_des)

    if curr_minus_ref:
        q_err = qconj(q_err)

    if q_err[0] < 0:
        q_err = -q_err

    q_err /= qnorm(q_err)
    axis, theta = qu_to_aa(q_err)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        axis = -axis

    return theta, axis
