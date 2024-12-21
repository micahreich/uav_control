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
from uav_control.planners.parametric_curve_planner import ParametricCurve

class ParametricCurveElement(PlotElement):
    def __init__(
        self,
        curve: ParametricCurve,
        curve_color="black",
        env: Optional[PlotEnvironment] = None,
    ):
        super().__init__(env, logging_level=LogLevel.ERROR)

        self.curve = curve
        self.curve_color = curve_color

    def init_environment(self, env):
        super().init_environment(env)

        duration = self.curve.duration
        n_pts = 500
        ts = np.linspace(0, duration, n_pts)

        xs = np.empty(shape=(n_pts, 3))

        for i in range(n_pts):
            xs[i] = self.curve.x(ts[i])

        self.env.ax.plot(xs[:, 0], xs[:, 1], xs[:, 2],
                            color=self.curve_color, linestyle="dashed",
                            lw=1.0)


if __name__ == "__main__":
    from uav_control.planners import parametric_curve_planner

    curve = parametric_curve_planner.ParametricCircle(duration=5.0,
                             radius=3.0,
                             center=np.array([5.0, 5.0, 5.0]),
                             normal=np.array([0.0, 0.0, 1.0])),

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")

    env = PlotEnvironment(fig, ax, sim_t_range=(0, 1), frame_rate=60)
    path_viz = ParametricCurveElement(curve)
    path_viz.init_environment(env)

    # env.ax.set_xlim([0, 10])
    # env.ax.set_ylim([0, 10])
    # env.ax.set_zlim([0, 10])
    plot_env = (
        PlotEnvironment(fig, ax, sim_t_range=[0, 1], frame_rate=1)
        .attach_element(
            ParametricCurveElement(curve)
        )
    )
    plot_env.render()
