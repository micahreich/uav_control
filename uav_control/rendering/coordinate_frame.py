from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from hybrid_ode_sim.simulation.base import BaseModel
from hybrid_ode_sim.simulation.rendering.base import (PlotElement,
                                                      PlotEnvironment)
from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from spatialmath.base import q2r, qconj, qslerp, qunit, r2q, rotz

from uav_control.constants import compose_state, decompose_state


class CoordinateFrame(PlotElement):
    def __init__(
        self,
        env: PlotEnvironment,
        system: Optional[BaseModel],
        show_origin=True,
        _debug_history: Optional[Tuple] = None,
    ):
        super().__init__(env, logging_level=LogLevel.ERROR)

        self.frame_x_vec = self.env.ax.quiver(0, 0, 0, 0, 0, 0, color="r", length=1.0)
        self.frame_y_vec = self.env.ax.quiver(0, 0, 0, 0, 0, 0, color="g", length=1.0)
        self.frame_z_vec = self.env.ax.quiver(0, 0, 0, 0, 0, 0, color="b", length=1.0)

        if show_origin:
            self.env.ax.scatter(0, 0, 0, color="gray", s=10, alpha=0.5)
            for v in np.eye(3):
                self.env.ax.quiver(
                    0,
                    0,
                    0,
                    *v,
                    color="black",
                    arrow_length_ratio=0.05,
                    lw=1.0,
                    alpha=0.5,
                )

        self.frame_x_text = self.env.ax.text(
            0,
            0,
            0,
            "x",
            color="r",
            fontsize=12,
            fontfamily="monospace",
            va="center",
            ha="center",
        )
        self.frame_y_text = self.env.ax.text(
            0,
            0,
            0,
            "y",
            color="g",
            fontsize=12,
            fontfamily="monospace",
            va="center",
            ha="center",
        )
        self.frame_z_text = self.env.ax.text(
            0,
            0,
            0,
            "z",
            color="b",
            fontsize=12,
            fontfamily="monospace",
            va="center",
            ha="center",
        )

        if _debug_history is None:
            self.ts, self.ys, self.history = system.history()
        else:
            assert len(_debug_history) == 3 and system is None
            self.ts, self.ys, self.history = _debug_history

    def set_text_location(self, text_locations):
        for i, (text, axis) in enumerate(
            zip(
                [self.frame_x_text, self.frame_y_text, self.frame_z_text],
                ["y", "x", "x"],
            )
        ):
            text.set_position((text_locations[0, i], text_locations[1, i]))
            text.set_3d_properties(text_locations[2, i], axis)

    def update(self, t):
        try:
            r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(self.history(t))
            rot_NB = q2r(q_NB)
        except Exception as e:
            self.logger.error(f"Could not decompose state: {e}")
            raise e

        x_vec_endpt = r_b0_N + rot_NB[:, 0]
        y_vec_endpt = r_b0_N + rot_NB[:, 1]
        z_vec_endpt = r_b0_N + rot_NB[:, 2]

        self.frame_x_vec.set_segments([[r_b0_N, x_vec_endpt]])
        self.frame_y_vec.set_segments([[r_b0_N, y_vec_endpt]])
        self.frame_z_vec.set_segments([[r_b0_N, z_vec_endpt]])

        text_locations = (
            rot_NB @ np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]).T
        ) + r_b0_N[:, None]
        self.set_text_location(text_locations)

        return (
            self.frame_x_vec,
            self.frame_y_vec,
            self.frame_z_vec,
            self.frame_x_text,
            self.frame_y_text,
            self.frame_z_text,
        )


if __name__ == "__main__":
    import matplotlib as mpl
    from spatialmath.base import eul2r, qvmul

    # Setting up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ts = np.array([0, 1, 2, 3, 4], dtype=float)
    orientations = np.array(
        [
            rotz(0),
            rotz(np.pi / 2),
            rotz(np.pi),
            rotz(3 * np.pi / 2),
            rotz(0),
            # eul2r(*np.random.rand(3) * 2 * np.pi) for _ in range(len(ts))
        ]
    )

    ys = np.array(
        [
            compose_state(r_b0_N=R @ np.array([1.0, 0.0, 0.0]), q_NB=r2q(R))
            for R in orientations
        ]
    )

    def system_history_fn(t):
        t_idx = np.searchsorted(ts, t)
        i_prev = max(0, t_idx - 1)
        i_next = min(len(ts) - 1, t_idx)

        if i_prev == i_next:
            return ys[i_prev]

        scale_prev = (ts[i_next] - t) / (ts[i_next] - ts[i_prev])
        scale_next = 1.0 - scale_prev

        r_b0_N_prev, q_NB_prev, v_b0_N_prev, omega_b0_B_prev = decompose_state(
            ys[i_prev]
        )
        r_b0_N_next, q_NB_next, v_b0_N_next, omega_b0_B_next = decompose_state(
            ys[i_next]
        )
        qnm = qunit(qslerp(q_NB_prev, q_NB_next, s=scale_next, shortest=True))

        return compose_state(r_b0_N=qvmul(qnm, np.array([1, 0, 0])), q_NB=qnm)

    env = PlotEnvironment(fig, ax, sim_t_range=[0.0, 4.0], frame_rate=60)
    coordinate_frame = CoordinateFrame(
        env, system=None, _debug_history=(ts, ys, system_history_fn)
    )

    env.render(plot_elements=[coordinate_frame])
