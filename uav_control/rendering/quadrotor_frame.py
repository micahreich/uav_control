from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from hybrid_ode_sim.simulation.base import BaseModel
from hybrid_ode_sim.simulation.rendering.base import (PlotElement,
                                                      PlotEnvironment)
from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from spatialmath.base import q2r, qconj, qslerp, qunit, r2q, rotx, roty, rotz

from uav_control.constants import compose_state, decompose_state


class QuadrotorFrame(PlotElement):
    def __init__(
        self,
        system: Optional[BaseModel],
        color="black",
        alpha=1.0,
        env: Optional[PlotEnvironment] = None,
        _debug_history: Optional[Tuple] = None,
    ):
        super().__init__(env, logging_level=LogLevel.ERROR)
        self.system = system
        self.color = color
        self.alpha = alpha

        if _debug_history is None:
            _, _, self.history = system.history()
        else:
            assert len(_debug_history) == 3 and system is None
            _, _, self.history = _debug_history

    def init_environment(self, env):
        super().init_environment(env)

        L = 0.3
        R = rotz(np.pi / 4)
        # R_body = roty(np.pi/4)

        self.motor_positions = np.array([[L, 0, 0], [-L, 0, 0], [0, L, 0], [0, -L, 0]]).T
        self.rotated_motor_positions = R @ self.motor_positions
        # rotated_motor_positions_N = R_body @ rotated_motor_positions

        pts = 50
        theta = np.linspace(0, 2 * np.pi, pts)
        self.rotor_xs = 0.5 * L * np.cos(theta)
        self.rotor_ys = 0.5 * L * np.sin(theta)
        self.rotor_zs = 0.25 * L * np.ones(pts)

        (self.x_axis,) = self.env.ax.plot(
            self.rotated_motor_positions[0, :2],
            self.rotated_motor_positions[1, :2],
            self.rotated_motor_positions[2, :2],
            marker="o",
            markersize=3.0,
            color=self.color,
            alpha=self.alpha,
            linewidth=2.0,
        )

        (self.y_axis,) = self.env.ax.plot(
            self.rotated_motor_positions[0, 2:],
            self.rotated_motor_positions[1, 2:],
            self.rotated_motor_positions[2, 2:],
            marker="o",
            markersize=3.0,
            color=self.color,
            alpha=self.alpha,
            linewidth=2.0,
        )

        self.rotors_plots = [None] * 4
        self.rotors_coords = [None] * 4

        for i, motor_position in enumerate(self.rotated_motor_positions.T):
            i_rotor_xs = self.rotor_xs + motor_position[0]
            i_rotor_ys = self.rotor_ys + motor_position[1]
            i_rotor_zs = self.rotor_zs

            i_rotor_coords_B = np.array([i_rotor_xs, i_rotor_ys, i_rotor_zs])

            self.rotors_coords[i] = i_rotor_coords_B
            (rotor_plot,) = self.env.ax.plot(
                i_rotor_coords_B[0, :],
                i_rotor_coords_B[1, :],
                i_rotor_coords_B[2, :],
                color="gray",
                linewidth=1.0,
                alpha=self.alpha,
            )
            self.rotors_plots[i] = rotor_plot

    def set_plot_xyz(self, plot, x, y, z):
        plot.set_data(x, y)
        plot.set_3d_properties(z)

    def update(self, t):
        try:
            r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(self.history(t))
            rot_NB = q2r(q_NB)
        except Exception as e:
            self.logger.error(f"Could not decompose state: {e}")
            raise e

        rotated_motor_positions_N = rot_NB @ self.rotated_motor_positions

        self.set_plot_xyz(
            self.x_axis,
            r_b0_N[0] + rotated_motor_positions_N[0, :2],
            r_b0_N[1] + rotated_motor_positions_N[1, :2],
            r_b0_N[2] + rotated_motor_positions_N[2, :2],
        )

        self.set_plot_xyz(
            self.y_axis,
            r_b0_N[0] + rotated_motor_positions_N[0, 2:],
            r_b0_N[1] + rotated_motor_positions_N[1, 2:],
            r_b0_N[2] + rotated_motor_positions_N[2, 2:],
        )

        for i, _ in enumerate(self.rotated_motor_positions.T):
            i_rotor_coords_N = rot_NB @ self.rotors_coords[i]

            self.set_plot_xyz(
                self.rotors_plots[i],
                r_b0_N[0] + i_rotor_coords_N[0, :],
                r_b0_N[1] + i_rotor_coords_N[1, :],
                r_b0_N[2] + i_rotor_coords_N[2, :],
            )

        return self.x_axis, self.y_axis, self.rotors_plots


if __name__ == "__main__":
    import matplotlib as mpl
    from spatialmath.base import eul2r, qvmul

    # Setting up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ts = np.array([0, 1, 2, 3, 4], dtype=float)
    orientations = np.array(
        [
            rotx(0),
            roty(np.pi / 2),
            rotx(np.pi),
            roty(3 * np.pi / 2),
            rotz(0),
            # eul2r(*np.random.rand(3) * 2 * np.pi) for _ in range(len(ts))
        ]
    )

    ys = np.array(
        [compose_state(r_b0_N=R @ np.array([1.0, 0.0, 0.0]), q_NB=r2q(R)) for R in orientations]
    )

    def system_history_fn(t):
        t_idx = np.searchsorted(ts, t)
        i_prev = max(0, t_idx - 1)
        i_next = min(len(ts) - 1, t_idx)

        if i_prev == i_next:
            return ys[i_prev]

        scale_prev = (ts[i_next] - t) / (ts[i_next] - ts[i_prev])
        scale_next = 1.0 - scale_prev

        r_b0_N_prev, q_NB_prev, v_b0_N_prev, omega_b0_B_prev = decompose_state(ys[i_prev])
        r_b0_N_next, q_NB_next, v_b0_N_next, omega_b0_B_next = decompose_state(ys[i_next])
        qnm = qunit(qslerp(q_NB_prev, q_NB_next, s=scale_next, shortest=True))

        return compose_state(r_b0_N=qvmul(qnm, np.array([1, 0, 0])), q_NB=qnm)

    env = PlotEnvironment(fig, ax, sim_t_range=[0.0, 4.0], frame_rate=60)
    coordinate_frame = QuadrotorFrame(env, system=None, _debug_history=(ts, ys, system_history_fn))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    env.render(plot_elements=[coordinate_frame])
