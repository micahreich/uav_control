from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import sympy as sym
from hybrid_ode_sim.simulation.base import ContinuousTimeModel, DiscreteTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel
from scipy.interpolate import interp1d
from spatialmath.base import q2r, qconj, qdotb, qnorm, qslerp, qunit, qvmul, skewa

import uav_control.constants as constants
from uav_control.constants import (
    OMEGA_B0_B,
    Q_NB,
    R_B0_N,
    TAU_B0_B,
    THRUST,
    V_B0_N,
    a_g_N,
    compose_control,
    compose_state,
    compose_state_dot,
    decompose_control,
    decompose_state,
    e3,
    thrust_axis_B,
)
from uav_control.utils.math import (
    compute_cross_product_dot,
    compute_unit_vector_ddot,
    compute_unit_vector_dot,
    dqu_dq_jacobian,
    sym_Aq,
    sym_Gq,
    sym_H,
    sym_Lq,
    sym_Rq,
    sym_skewsym,
    vee,
)

p_dim, v_dim, omega_dim, tau_dim = 3, 3, 3, 3  # Position, velocity, angular velocity, torque
c_dim = 1  # Collective thrust
q_dim = 4  # Unit quaternion

nx = p_dim + v_dim + q_dim + omega_dim  # Number of states
nu = c_dim + tau_dim  # Number of controls


@dataclass
class QuadrotorRigidBodyParams:
    m: float = field(default=1.0)  # mass
    I: np.ndarray = field(default=np.eye(3))  # inertia matrix
    I_inv: np.ndarray = field(init=False)
    D_drag: np.ndarray = field(default=np.zeros(3))  # diagonal drag coefficient matrix

    def __post_init__(self):
        self.I_inv = np.linalg.inv(self.I)


@dataclass
class QuadrotorLinearization:
    Jx: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nx)))  # A matrix
    Ju: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx, nu)))  # B matrix
    x0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nx,)))  # linearization state
    u0: np.ndarray = field(default_factory=lambda: np.zeros(shape=(nu,)))  # linearization control


class QuadrotorRigidBodyDynamics(ContinuousTimeModel):
    @staticmethod
    def _symbolic_dynamics():
        I = sym.MatrixSymbol('I', 3, 3)  # Inertia matrix
        I_inv = sym.MatrixSymbol('I_inv', 3, 3)  # Inertia matrix
        g = sym.symbols('g')  # Gravitation constant
        m = sym.symbols('m')  # Mass

        p = sym.MatrixSymbol('p', p_dim, 1)  # Position (world frame)
        v = sym.MatrixSymbol('v', v_dim, 1)  # Velocity (world frame)
        omega = sym.MatrixSymbol('omega', omega_dim, 1)  # Angular velocity (body frame)
        q = sym.MatrixSymbol('q', q_dim, 1)  # Attitude (unit quaternion)

        x = sym.BlockMatrix([[p], [q], [v], [omega]]).as_explicit()

        tau = sym.MatrixSymbol('tau', tau_dim, 1)
        c = sym.symbols('c')

        u = sym.BlockMatrix([[sym.Matrix([c])], [tau]]).as_explicit()

        # Equations of motion
        pdot = v
        vdot = 1 / m * (sym.Matrix([0, 0, m * g]) + sym_Aq(q) @ sym.Matrix([0, 0, c]))
        qdot = 1 / 2 * sym_Lq(q) @ sym_H @ omega
        omegadot = I_inv @ (tau - sym_skewsym(omega) @ I @ omega)

        dx_dt = sym.BlockMatrix([[pdot], [qdot], [vdot], [omegadot]]).as_explicit()

        df_dxbar = dx_dt.jacobian(x)
        attitude_jac_dfdx = sym.BlockDiagMatrix(
            sym.eye(p_dim), sym_Gq(q), sym.eye(v_dim), sym.eye(omega_dim)
        ).as_explicit()

        df_dx = df_dxbar @ attitude_jac_dfdx
        df_du = dx_dt.jacobian(u)

        return dx_dt, df_dx, df_du, x, u

    dx_dt_symbolic, df_dx_symbolic, df_du_symbolic, x, u = _symbolic_dynamics.__func__()

    def __init__(
        self,
        y0: np.ndarray,
        params: QuadrotorRigidBodyParams,
        logging_level=LogLevel.ERROR,
    ):
        """
        Initializes the QuadrotorDynamics class.

        Parameters
        ----------
        y0 : np.ndarray
            The initial state of the quadrotor. This should be a 13x1 vector with the following elements:
            - r_b0_N: 3x1 vector, the position of the quadrotor in the ENU frame
            - q_NB: 4x1 vector, the attitude of the quadrotor as a (unit) quaternion
            - v_b0_N: 3x1 vector, the velocity of the quadrotor in the ENU frame
            - omega_b0_B: 3x1 vector, the angular velocity of the quadrotor in the body frame
        params : QuadrotorRigidBodyParams
            The parameters of the UAV rigid body model.
        logging_level : LogLevel, optional
            The logging level for the model. Defaults to LogLevel.ERROR.
        """
        super().__init__(y0, "quadrotor_state", params, logging_level=logging_level)

        (
            self.dx_dt_symbolic_paramified,
            self.df_dx_symbolic_paramified,
            self.df_du_symbolic_paramified,
        ) = self._paramify_symbolic_dynamics()

    def output_validate(self, y: np.ndarray) -> np.ndarray:
        """
        Validates the output of the quadrotor dynamics, ensuring unit-norm quaternions.

        Parameters
        ----------
        y : np.ndarray
            The output to be validated.

        Returns
        -------
        np.ndarray
            The validated output.
        """

        r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(y)

        return compose_state(r_b0_N, q_NB / qnorm(q_NB), v_b0_N, omega_b0_B)

    def _paramify_symbolic_dynamics(self):
        I = sym.MatrixSymbol('I', 3, 3)  # Inertia matrix
        I_inv = sym.MatrixSymbol('I_inv', 3, 3)  # Inertia matrix
        g = sym.symbols('g')  # Gravitation constant
        m = sym.symbols('m')  # Mass

        p = sym.MatrixSymbol('p', p_dim, 1)  # Position (world frame)
        v = sym.MatrixSymbol('v', v_dim, 1)  # Velocity (world frame)
        omega = sym.MatrixSymbol('omega', omega_dim, 1)  # Angular velocity (body frame)
        q = sym.MatrixSymbol('q', q_dim, 1)  # Attitude (unit quaternion)

        parameter_substituions = {
            I: sym.Matrix(self.params.I),
            I_inv: sym.Matrix(self.params.I_inv),
            m: self.params.m,
            g: constants.g,
        }

        dx_dt_paramified = QuadrotorRigidBodyDynamics.dx_dt_symbolic.subs(parameter_substituions)
        df_dx_paramified = QuadrotorRigidBodyDynamics.df_dx_symbolic.subs(parameter_substituions)
        df_du_paramified = QuadrotorRigidBodyDynamics.df_du_symbolic.subs(parameter_substituions)

        dx_dt_symbolic_paramified = sym.lambdify(
            (QuadrotorRigidBodyDynamics.x, QuadrotorRigidBodyDynamics.u), dx_dt_paramified, 'numpy'
        )

        df_dx_symbolic_paramified = sym.lambdify(
            (QuadrotorRigidBodyDynamics.x, QuadrotorRigidBodyDynamics.u), df_dx_paramified, 'numpy'
        )

        df_du_symbolic_paramified = sym.lambdify(
            (QuadrotorRigidBodyDynamics.x, QuadrotorRigidBodyDynamics.u), df_du_paramified, 'numpy'
        )

        return dx_dt_symbolic_paramified, df_dx_symbolic_paramified, df_du_symbolic_paramified

    def continuous_dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculates the continuous dynamics of the quadrotor.

        Parameters
        ----------
        t : float
            The current time.
        y : np.ndarray
            The state vector of the quadrotor.

        Returns
        -------
        np.ndarray
            The derivative of the state vector.

        Notes
        -----
        Computes the rigid body dynamics (state derivatives) of a quadrotor UAV based on a first-order drag model.
        """

        try:
            allocated_wrench = self.input_models["allocator"].y
        except KeyError:
            self.logger.warning(
                "No input model found for 'allocator'. Defaulting to zero thrust and torque."
            )
            allocated_wrench = np.zeros(4)

        return self.dx_dt_symbolic_paramified(y, allocated_wrench)

    def history(self) -> Tuple[np.ndarray, np.ndarray, interp1d]:
        """
        Returns the history of the quadrotor's state over time.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, interp1d]
            - Array of time points
            - Array of state vectors
            - Interpolation function for the state
        """
        ts = np.array(self.t_history)
        ys = np.array(self.y_history)

        def interpolator_fn(t):
            assert (
                t >= ts[0] and t <= ts[-1]
            ), f"Time {t} is outside the range of the simulation history."

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
                omega_b0_B=omega_b0_B_prev * scale_prev + omega_b0_B_next * scale_next,
            )

        return ts, ys, interpolator_fn

    def compute_differential_flatness_states_controls(
        self,
        r_b0_N_ref,
        v_b0_N_ref,
        a_b0_N_ref,
        j_b0_N_ref,
        s_b0_N_ref,
        b_1d,
        b_1d_dot,
        b_1d_ddot,
    ):
        """
        Compute the quadrotor state and controls as a function of the differentially flat
        outputs, ie position and desired first-body axis (and their time derivatives).

        Parameters
        ----------
        params : QuadrotorRigidBodyParams
            The parameters of the quadrotor rigid body model
        r_b0_N_ref : np.ndarray (3,)
            The reference position of the quadrotor CoM in the world frame
        v_b0_N_ref : np.ndarray (3,)
            The reference velocity of the quadrotor CoM in the world frame
        a_b0_N_ref : np.ndarray (3,)
            The reference acceleration of the quadrotor CoM in the world frame
        j_b0_N_ref : np.ndarray (3,)
            The reference jerk of the quadrotor CoM in the world frame
        s_b0_N_ref : np.ndarray (3,)
            The reference snap of the quadrotor CoM in the world frame
        b_1d : np.ndarray (3,)
            The desired first body-fixed axis of the quadrotor
        b_1d_dot : np.ndarray (3,)
            The time derivative of the desired first body-fixed axis of the quadrotor
        b_1d_ddot : np.ndarray (3,)
            The second time derivative of the desired first body-fixed axis of the quadrotor

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
            The quadrotor position, velocity, orientation (reference frame), and angular velocity (expressed in reference frame);
            The desired thrust vector and angular acceleration (expressed in reference frame)
        """

        a_b0_N_adjusted = a_b0_N_ref - a_g_N  # TODO: add drag model
        f_des = self.params.m * a_b0_N_adjusted  # Desired thrust vector, gravity-compensated

        # Construct desired/commanded orientation of the body frame
        b_3c = a_b0_N_adjusted / np.linalg.norm(a_b0_N_adjusted)  # Desired body z-axis

        C = np.cross(b_3c, b_1d)
        b_2c = C / np.linalg.norm(C)

        b_1c = np.cross(b_2c, b_3c)

        rot_ND = np.column_stack((b_1c, b_2c, b_3c))  # Rotation from desired frame to world

        # Construct desired/commanded angular velocity of the body frame
        b_3c_dot = compute_unit_vector_dot(a_b0_N_adjusted, j_b0_N_ref)

        C_dot = compute_cross_product_dot(b_3c, b_3c_dot, b_1d, b_1d_dot)
        b_2c_dot = compute_unit_vector_dot(C, C_dot)

        b_1c_dot = np.cross(b_2c_dot, b_3c) + np.cross(b_2c, b_3c_dot)

        rot_ND_dot = np.column_stack((b_1c_dot, b_2c_dot, b_3c_dot))
        omega_d0_D_skewa = rot_ND.T @ rot_ND_dot
        omega_d0_D = vee(omega_d0_D_skewa)

        # Construct desired/commanded angular acceleration of the body frame
        b_3c_ddot = compute_unit_vector_ddot(a_b0_N_adjusted, j_b0_N_ref, s_b0_N_ref)

        C_ddot = compute_cross_product_dot(
            b_3c_dot, b_3c_ddot, b_1d, b_1d_dot
        ) + compute_cross_product_dot(b_3c, b_3c_dot, b_1d_dot, b_1d_ddot)

        b_2c_ddot = compute_unit_vector_ddot(C, C_dot, C_ddot)

        b_1c_ddot = compute_cross_product_dot(
            b_2c_dot, b_2c_ddot, b_3c, b_3c_dot
        ) + compute_cross_product_dot(b_2c, b_2c_dot, b_3c_dot, b_3c_ddot)

        rot_ND_ddot = np.column_stack((b_1c_ddot, b_2c_ddot, b_3c_ddot))
        omega_d0_D_dot = vee(rot_ND.T @ rot_ND_ddot - omega_d0_D_skewa @ omega_d0_D_skewa)

        return [r_b0_N_ref, v_b0_N_ref, rot_ND, omega_d0_D], [f_des, omega_d0_D_dot]


if __name__ == "__main__":
    print('thrust', THRUST)
    # x = sym.MatrixSymbol('x', 3, 1)
    # y = sym.MatrixSymbol('y', 3, 1)

    # xy_stacked = sym.BlockMatrix([
    #     [x],
    #     [y]
    # ]).as_explicit()

    # expr = x + y
    # expr2 = expr.subs({x: xy_stacked[[0, 1, 2], 0], y: xy_stacked[3:6, 0]})
    # expr2_app = sym.lambdify((xy_stacked), expr2, 'numpy')

    # # print(expr2_app(np.array([1, 2, 3, 4, 5, 6])))
    # vv = np.array([1, 2, 3]).reshape((3,1))
    # print( expr2_app(np.array([1, 2, 3, 0, 100, 0])) )

    params = QuadrotorRigidBodyParams()
    quad = QuadrotorRigidBodyDynamics(compose_state(), params)

    dxdt = quad.dx_dt_symbolic

    I = sym.MatrixSymbol('I', 3, 3)  # Inertia matrix
    I_inv = sym.MatrixSymbol('I_inv', 3, 3)  # Inertia matrix
    g = sym.symbols('g')  # Gravitation constant
    m = sym.symbols('m')  # Mass

    p = sym.MatrixSymbol('p', p_dim, 1)  # Position (world frame)
    v = sym.MatrixSymbol('v', v_dim, 1)  # Velocity (world frame)
    omega = sym.MatrixSymbol('omega', omega_dim, 1)  # Angular velocity (body frame)
    q = sym.MatrixSymbol('q', q_dim, 1)  # Attitude (unit quaternion)

    tau = sym.MatrixSymbol('tau', tau_dim, 1)
    c = sym.symbols('c')

    parameter_substituions = {
        I: sym.Matrix(params.I),
        I_inv: sym.Matrix(params.I_inv),
        m: params.m,
        g: constants.g,
    }
    dx_dt_paramified = QuadrotorRigidBodyDynamics.dx_dt_symbolic.subs(parameter_substituions)

    # print(dx_dt_paramified)
    print(QuadrotorRigidBodyDynamics.x.shape, QuadrotorRigidBodyDynamics.u)

    dx_dt_symbolic_paramified = sym.lambdify(
        (QuadrotorRigidBodyDynamics.x, QuadrotorRigidBodyDynamics.u), dx_dt_paramified, 'numpy'
    )

    xx, uu = np.zeros(nx), np.zeros(nu)
    # xx = xx.reshape((-1,1))
    # uu = uu.reshape((-1,1))

    print(xx.shape, uu.shape)
    print(dx_dt_paramified.shape)

    # print(dx_dt_paramified.subs(v, sym.Matrix(np.zeros(3))))

    # [p0, q0, v0, omega0] = decompose_state(xx)
    # [c0, tau0] = decompose_control(uu)

    # # add column
    # p0 = p0.reshape((-1,1))
    # q0 = q0.reshape((-1,1))
    # v0 = v0.reshape((-1,1))
    # omega0 = omega0.reshape((-1,1))
    # tau0 = tau0.reshape((-1,1))

    print(dx_dt_symbolic_paramified(xx, uu))

    # test_instance = Test()
    # print(test_instance.x)  # This will print 5
    # print(Test.x)  # Accessing the class variable directly
