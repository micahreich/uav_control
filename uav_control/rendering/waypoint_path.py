import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from spatialmath.base import q2r, r2q, rotz, qunit, qslerp
from uav_control.constants import decompose_state, compose_state
from hybrid_ode_sim.simulation.rendering.base import PlotEnvironment, PlotElement
from typing import Tuple, Optional


class WaypointTrajectory(PlotElement):
    def __init__(self, env: PlotEnvironment, waypoints,
                 waypoint_color='black'):
        super().__init__(env)
        
        text_expansion = 0.35
        
        for i, w in enumerate(waypoints):
            self.env.ax.text(w[0]+text_expansion, w[1]+text_expansion, w[2]+text_expansion, f"{i}", color=waypoint_color, fontsize=10, fontfamily='monospace', va='center', ha='center')
            self.env.ax.scatter(w[0], w[1], w[2], color=waypoint_color, alpha=0.6, s=10)
        
if __name__ == "__main__":    
    waypoints=np.array([
        [0.0, 0.0, 1.0],
        [0.0, 5.0, 5.0]
    ])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    env = PlotEnvironment(fig, ax, sim_t_range=(0, 1), frame_rate=60)
    path_viz = WaypointTrajectory(env, waypoints)
    
    env.ax.set_xlim([0, 10])
    env.ax.set_ylim([0, 10])
    env.ax.set_zlim([0, 10])
    
    env.render(plot_elements=[path_viz])
