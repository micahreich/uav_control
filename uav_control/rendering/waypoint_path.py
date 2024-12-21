from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from hybrid_ode_sim.simulation.rendering.base import (PlotElement,
                                                      PlotEnvironment)
from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from spatialmath.base import q2r, qslerp, qunit, r2q, rotz

from uav_control.constants import compose_state, decompose_state


class WaypointTrajectoryElement(PlotElement):
    def __init__(
        self,
        waypoints,
        waypoint_color="black",
        env: Optional[PlotEnvironment] = None,
    ):
        super().__init__(env, logging_level=LogLevel.ERROR)

        self.text_expansion = 0.35
        self.waypoints = waypoints
        self.waypoint_color = waypoint_color

    def init_environment(self, env):
        super().init_environment(env)

        for i, w in enumerate(self.waypoints):
            self.env.ax.text(
                w[0] + self.text_expansion,
                w[1] + self.text_expansion,
                w[2] + self.text_expansion,
                f"{i}",
                color=self.waypoint_color,
                fontsize=10,
                fontfamily="monospace",
                va="center",
                ha="center",
            )

            self.env.ax.scatter(w[0], w[1], w[2], color=self.waypoint_color, alpha=0.6, s=10)


if __name__ == "__main__":
    waypoints = np.array([[0.0, 0.0, 1.0], [0.0, 5.0, 5.0]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    env = PlotEnvironment(fig, ax, sim_t_range=(0, 1), frame_rate=60)
    path_viz = WaypointTrajectoryElement(env, waypoints)

    env.ax.set_xlim([0, 10])
    env.ax.set_ylim([0, 10])
    env.ax.set_zlim([0, 10])

    env.render(plot_elements=[path_viz])
