from dataclasses import dataclass
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

from hybrid_ode_sim.simulation.base import ContinuousTimeModel, DiscreteTimeModel
from hybrid_ode_sim.simulation.ode_solvers.adaptive_step_solver import RK23
from hybrid_ode_sim.simulation.ode_solvers.fixed_step_solver import RK4
from hybrid_ode_sim.simulation.rendering.base import PlotEnvironment
from hybrid_ode_sim.simulation.rendering.pose3d import Pose3D
from hybrid_ode_sim.simulation.simulator import (
    ModelGraph,
    Simulator,
    SimulationEnvironment,
)
from hybrid_ode_sim.utils.logging_tools import LogLevel

from spatialmath.base import qdotb, qunit, qvmul, qeye

class MovingCoordinateFrame(ContinuousTimeModel):
    def __init__(self, y0, name = "moving_coordinate_frame", params=None, logging_level=LogLevel.ERROR):
        super().__init__(y0, name, params, logging_level)
        self.path_time = 5
        self.R = 1.0

    def output_validate(self, y):
        r, q = y[:3], y[3:]
        return np.concatenate((r, qunit(q)))

    def continuous_dynamics(self, t, y) -> np.ndarray:
        r, q = y[:3], y[3:]
        omega = qvmul(q, np.array([1.0, 1.0, 1.0]))

        v = np.array([
            -self.R * 2 * np.pi * (1/self.path_time) * np.sin(2 * np.pi * (1/self.path_time) * t),
            self.R * 2 * np.pi * (1/self.path_time) * np.cos(2 * np.pi * (1/self.path_time) * t),
            0.0
        ])

        return np.concatenate((v, qdotb(y[3:], omega)))

if __name__ == "__main__":
    r0, q0 = np.array([1.0, 0, 0]), qeye()
    moving_frame = MovingCoordinateFrame(y0=np.concatenate((r0, q0)))
    model_graph = ModelGraph(models=[moving_frame])

    # Set up simulator
    t_range = [0, 5]
    simulator = Simulator(model_graph, RK4(h=0.01))

    # Set up rendering
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    plot_env = PlotEnvironment(
        fig, ax, t_range, frame_rate=30, t_start=0.0
    ).attach_element(
        Pose3D(system=moving_frame)
    )

    env = SimulationEnvironment(simulator=simulator, plot_env=plot_env).run_simulation(
        show_time=True,
        realtime=True,
        t_range=t_range,
    )
