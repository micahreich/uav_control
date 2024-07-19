from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from hybrid_ode_sim.simulation.rendering.base import (PlotElement,
                                                      PlotEnvironment)
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from spatialmath.base import q2r, qslerp, qunit, r2q, rotz

from uav_control.constants import compose_state, decompose_state


class PolynomialTrajectory(PlotElement):
    def __init__(
        self,
        env: PlotEnvironment,
        polytraj,
        path_t_eval_range: Optional[Tuple[float, float]] = None,
        color_velocities=True,
        plot_waypoints=True,
    ):
        super().__init__(env)

        if path_t_eval_range is None:
            ts = np.linspace(*self.env.t_range, polytraj.n_segments * 150)
        else:
            ts = np.linspace(*path_t_eval_range, polytraj.n_segments * 150)

        positions = np.empty(shape=(len(ts), 3))
        velocity_norms = np.empty(shape=(len(ts),))

        for i, t in enumerate(ts):
            r, v = polytraj(t, n_derivatives=1)
            positions[i] = r
            velocity_norms[i] = np.linalg.norm(v)

        if color_velocities:
            # Define 3D points and segments
            points = positions.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a continuous norm to map from data points to colors
            norm = Normalize(velocity_norms.min(), velocity_norms.max())
            lc = Line3DCollection(
                segments, linewidths=1.0, cmap="gist_rainbow", norm=norm
            )

            # Set the values used for colormapping
            lc.set_array(velocity_norms)
            lc.set_linewidth(2)

            # Add the collection to the plot
            self.env.ax.add_collection(lc)

            # Adding color bar
            cbar = self.env.fig.colorbar(lc, ax=self.env.ax, pad=0.1)
            cbar.set_label("Velocity [m/s]")
        else:
            self.env.ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color="gray",
                linestyle="dashed",
                lw=1.0,
            )

        # Plot waypoints of the trajectory
        if plot_waypoints:
            self.env.ax.scatter(
                polytraj.waypoints[:, 0],
                polytraj.waypoints[:, 1],
                polytraj.waypoints[:, 2],
                color="black",
                alpha=0.6,
                s=10,
            )


if __name__ == "__main__":
    from hybrid_ode_sim.systems.polynomial_trajgen import (
        PolynomialTrajectoryND, visualize_spatial_trajectory)

    # n_points = 4
    # times = np.arange(n_points, dtype=np.float64)
    # waypoints = np.random.rand(n_points, 3) * 10
    # traj = PolynomialTrajectoryND(waypoints, times, minimize_order=4)
    # # fig = visualize_spatial_trajectory(traj)
    # # plt.show()
    # Straight Line Trajectory
    n_points = 2

    times = np.array([0.0, 5.0])
    traj = PolynomialTrajectoryND(
        waypoints=np.array([[0.0, 0.0, 1.0], [0.0, 5.0, 5.0]]),
        timepoints=times,
        minimize_order=4,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    env = PlotEnvironment(fig, ax, sim_t_range=(times[0], times[-1]), frame_rate=60)
    path_viz = PolynomialTrajectory(env, traj, color_velocities=True)

    env.ax.set_xlim([0, 10])
    env.ax.set_ylim([0, 10])
    env.ax.set_zlim([0, 10])

    env.render(plot_elements=[path_viz])
