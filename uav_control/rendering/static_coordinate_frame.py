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
from spatialmath import SE3
from uav_control.constants import compose_state, decompose_state


class StaticCoordinateFrameElement(PlotElement):
    def __init__(
        self,
        pose: SE3,
        env: Optional[PlotEnvironment] = None,
    ):
        super().__init__(env, logging_level=LogLevel.ERROR)
        self.pose = pose

    def init_environment(self, env):
        super().init_environment(env)

        r_b0_N = self.pose.t
        rot_NB = self.pose.R

        x_vec_endpt = rot_NB[:, 0]
        y_vec_endpt = rot_NB[:, 1]
        z_vec_endpt = rot_NB[:, 2]

        self.env.ax.scatter(*r_b0_N, color="black", s=12)

        self.frame_x_vec = self.env.ax.quiver(*r_b0_N, *x_vec_endpt, color="black", length=1.0, alpha=0.5)
        self.frame_y_vec = self.env.ax.quiver(*r_b0_N, *y_vec_endpt, color="black", length=1.0, alpha=0.5)
        self.frame_z_vec = self.env.ax.quiver(*r_b0_N, *z_vec_endpt, color="black", length=1.0, alpha=0.5)

        text_locations = (rot_NB @ np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]).T) + r_b0_N[
            :, None
        ]

        self.frame_x_text = self.env.ax.text(
            *text_locations[:, 0],
            "x",
            color="black",
            fontsize=12,
            fontfamily="monospace",
            va="center",
            ha="center",
            alpha=0.5,
        )
        self.frame_y_text = self.env.ax.text(
            *text_locations[:, 1],
            "y",
            color="black",
            fontsize=12,
            fontfamily="monospace",
            va="center",
            ha="center",
            alpha=0.5,
        )
        self.frame_z_text = self.env.ax.text(
            *text_locations[:, 2],
            "z",
            color="black",
            fontsize=12,
            fontfamily="monospace",
            va="center",
            ha="center",
            alpha=0.5,
        )

if __name__ == "__main__":
    import matplotlib as mpl
    from spatialmath.base import eul2r, qvmul

    # Setting up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    plot_env = PlotEnvironment(fig, ax, sim_t_range=(0, 1), frame_rate=20)
    plot_env.attach_element(
        StaticCoordinateFrameElement(
            pose=SE3.Rand(xrange=[-1, 1], yrange=[-1, 1], zrange=[-1, 1])
        )
    )

    plot_env.render(show_time=False)
