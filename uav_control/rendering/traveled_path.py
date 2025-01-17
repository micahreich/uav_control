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


class TraveledPathElement(PlotElement):
    def __init__(
        self, system: BaseModel, color="blue", alpha=0.5, linewidth=1, env: Optional[PlotEnvironment] = None
    ):
        super().__init__(env, logging_level=LogLevel.ERROR)

        self.system = system
        self.color = color
        self.alpha = alpha
        self.linewidth = linewidth

        _, _, self.history = system.history()

    def init_environment(self, env):
        super().init_environment(env)

        (self.trail_plot,) = self.env.ax.plot(
            [], [], [], linewidth=self.linewidth, color=self.color, alpha=self.alpha
        )
        self.x_history, self.y_history, self.z_history = [], [], []

    def set_plot_xyz(self, plot, x, y, z):
        plot.set_data(x, y)
        plot.set_3d_properties(z)

    def update(self, t):
        try:
            r_b0_N, q_NB, v_b0_N, omega_b0_B = decompose_state(self.history(t))

            self.x_history.append(r_b0_N[0])
            self.y_history.append(r_b0_N[1])
            self.z_history.append(r_b0_N[2])

            rot_NB = q2r(q_NB)
        except Exception as e:
            self.logger.error(f"Could not decompose state: {e}")
            raise e

        self.set_plot_xyz(self.trail_plot, self.x_history, self.y_history, self.z_history)

        return (self.trail_plot,)

    def reset(self):
        self.x_history, self.y_history, self.z_history = [], [], []
        self.set_plot_xyz(self.trail_plot, [], [], [])

        return (self.trail_plot,)
