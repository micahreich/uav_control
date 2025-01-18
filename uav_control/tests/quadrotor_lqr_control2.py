import os

import matplotlib.pyplot as plt
import numpy as np
import spatialmath as sm
from hybrid_ode_sim.simulation.ode_solvers.fixed_step_solver import RK4
from hybrid_ode_sim.simulation.rendering.base import PlotEnvironment
from hybrid_ode_sim.simulation.simulator import (ModelGraph,
                                                 SimulationEnvironment,
                                                 Simulator)
from hybrid_ode_sim.utils.logging_tools import LogLevel

from uav_control.allocators.qp_control_allocator import (
    QuadrotorQPAllocator, QuadrotorQPAllocatorParams)
from uav_control.constants import (OMEGA_B0_B, Q_NB, R_B0_N, V_B0_N,
                                   compose_state, g)
from uav_control.controllers.lqr_controller import (LQRController,
                                                    LQRControllerParams,
                                                    LQRDynamicsLinearization,
                                                    BodyrateController,
                                                    BodyrateControllerParams)
from uav_control.dynamics import (QuadrotorRigidBodyDynamics,
                                  QuadrotorRigidBodyParams,
                                  QuadrotorDisturbanceWrench,
                                  QuadrotorDisturbanceWrenchParams)
from uav_control.planners.differential_flatness_planner import \
    DifferentialFlatnessPlanner
from uav_control.planners.parametric_curve_planner import ParametricCircle, QuadrotorParametricCurvePlanner
from uav_control.planners.point_stabilize_planner import (
    QuadrotorStabilizationPlanner, QuadrotorStabilizationPlannerParams)
from uav_control.planners.polynomial_planner import (
    QuadrotorPolynomialPlanner, QuadrotorPolynomialPlannerParams)
from uav_control.planners.waypoint_planner import (
    QuadrotorWaypointPlanner, QuadrotorWaypointPlannerParams)
from uav_control.rendering.coordinate_frame import CoordinateFrameElement
from uav_control.rendering.parametric_curve import ParametricCurveElement
from uav_control.rendering.polynomial_path import PolynomialTrajectoryElement
from uav_control.rendering.quadrotor_frame import QuadrotorFrameElement
from uav_control.rendering.traveled_path import TraveledPathElement
from uav_control.rendering.waypoint_path import WaypointTrajectoryElement
from uav_control.utils.find_package_root import find_package_root

SW_SAMPLE_RATE = 50  # Hz
LINEARIZATION_RATE = 10  # Hz

M = 1.1
I = np.diag([0.0112, 0.01123, 0.02108])
L = 0.315
D_drag = 0.0 * np.diag([0.26, 0.28, 0.42])
cq_ct_ratio = 8.004e-2


def hover_stabilize():
    point_stabilization_target = QuadrotorStabilizationPlanner(
        y0=0,
        sample_rate=10,
        params=QuadrotorStabilizationPlannerParams(
            position=np.array([2.0, 1.0, 0.0]),
            b1d=np.array([1.0, 0.0, 0.0])
        ),
    )

    dfb_planner = DifferentialFlatnessPlanner(
        sample_rate=SW_SAMPLE_RATE,
        planner_name="point_stabilize_planner",
    )

    rigid_body_params = QuadrotorRigidBodyParams(m=M, I=I, D_drag=D_drag)

    # Create quadrotor and disturbance
    quadrotor = QuadrotorRigidBodyDynamics(
        y0=compose_state(
            r_b0_N=np.array([0.0, 0.0, 1.0]),
            q_NB=sm.base.r2q(sm.base.rotx(20, "deg")),
        ),
        params=rigid_body_params,
        logging_level=LogLevel.ERROR,
        noise_y0=True,
    )

    disturbance = QuadrotorDisturbanceWrench(
        sample_rate=SW_SAMPLE_RATE,
        params=QuadrotorDisturbanceWrenchParams(
            force_N=np.array([5.0, 0.0, 0.0]),
            torque_N=np.zeros(3),
            active_interval=(2.0, 2.5)
        ),
    )

    # Create motor allocator
    allocator = QuadrotorQPAllocator(
        y0=np.array([-rigid_body_params.m * g, 0.0, 0.0, 0.0]),
        sample_rate=SW_SAMPLE_RATE,
        params=QuadrotorQPAllocatorParams(
            W=np.diag([0.001, 10.0, 10.0, 0.1]),
            l=L,
            u_min=-np.inf,
            u_max=np.inf,
            cq_ct_ratio=cq_ct_ratio,
        ),
    )

    linearization = LQRDynamicsLinearization(
        sample_rate=LINEARIZATION_RATE,
        params=LQRControllerParams.from_weights(
            r_weights=10 * np.ones(3),
            aa_weights=np.ones(3),
            v_weights=0.1 * np.ones(3),
            omega_weights=0.1 * np.array([10, 10, 100]),
            collective_thrust_weight=0.1,
        ),
    )

    lqr_controller = LQRController(
        sample_rate=SW_SAMPLE_RATE,
    )

    bodyrate_controller = BodyrateController(
        y0 = np.zeros(4),
        sample_rate=SW_SAMPLE_RATE,
        params=BodyrateControllerParams(
            P = 20 * np.eye(3),
        ),
    )

    # Create simulator
    point_stabilization_target.inputs_to(dfb_planner)
    dfb_planner.inputs_to(linearization)
    linearization.inputs_to(lqr_controller)
    lqr_controller.inputs_to(bodyrate_controller)
    bodyrate_controller.inputs_to(allocator)
    allocator.inputs_to(quadrotor)
    disturbance.inputs_to(quadrotor)

    dfb_planner.feedback_from(quadrotor)

    linearization.feedback_from(quadrotor)
    linearization.feedback_from(bodyrate_controller)

    lqr_controller.feedback_from(quadrotor)
    lqr_controller.feedback_from(linearization)

    bodyrate_controller.feedback_from(quadrotor)
    bodyrate_controller.feedback_from(lqr_controller)

    model_graph = ModelGraph(
        models=[
            point_stabilization_target,
            dfb_planner,
            linearization,
            lqr_controller,
            bodyrate_controller,
            allocator,
            quadrotor,
            # disturbance
        ]
    )

    t_range = [0, 8.0]
    simulator = Simulator(model_graph, RK4(h=0.1))

    # Create visualization environment
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_zlim(-1, 4)

    plot_env = (
        PlotEnvironment(fig, ax, t_range, frame_rate=20, t_start=0.0)
        .attach_element(
            QuadrotorFrameElement(
                system=quadrotor,
                color="black",
                alpha=1.0,
            )
        )
        .attach_element(
            TraveledPathElement(
                system=quadrotor,
                linewidth=1.0
            )
        )
    )

    # Run simulation
    env = SimulationEnvironment(
        simulator=simulator,
        plot_env=plot_env,
    ).run_simulation(t_range=t_range, realtime=False, show_time=True)


def circle_follow():
    t_range = [0, 10.0]

    curve = ParametricCircle(duration=4.0,
                             radius=2.0,
                             center=np.array([0.0, 0.0, 1.0]),
                             normal=np.array([0.5, 0.0, 1.0]))

    curve_planner = QuadrotorParametricCurvePlanner(
        y0=None,
        sample_rate=10,
        curve=curve,
        name="circle_planner",
        static_heading=True,
    )

    dfb_planner = DifferentialFlatnessPlanner(
        sample_rate=SW_SAMPLE_RATE,
        planner_name="circle_planner",
    )

    rigid_body_params = QuadrotorRigidBodyParams(m=M, I=I, D_drag=D_drag)

    quadrotor = QuadrotorRigidBodyDynamics(
        y0=compose_state(
            r_b0_N=curve.x(t_range[0]), #+ np.random.normal(loc=0.5, size=3),
            q_NB=sm.base.r2q(sm.base.rotz(0, "deg")),
        ),
        params=rigid_body_params,
        logging_level=LogLevel.ERROR,
        noise_y0=True,
    )

    # Create motor allocator
    allocator = QuadrotorQPAllocator(
        y0=np.array([-rigid_body_params.m * g, 0.0, 0.0, 0.0]),
        sample_rate=SW_SAMPLE_RATE,
        params=QuadrotorQPAllocatorParams(
            W=np.diag([0.001, 10.0, 10.0, 0.1]),
            l=L,
            u_min=-np.inf,
            u_max=np.inf,
            cq_ct_ratio=cq_ct_ratio,
        ),
    )

    linearization = LQRDynamicsLinearization(
        sample_rate=LINEARIZATION_RATE,
        params=LQRControllerParams.from_weights(
            r_weights=np.ones(3),
            aa_weights=np.ones(3),
            v_weights=0.1 * np.ones(3),
            omega_weights=0.1 * np.ones(3),
            collective_thrust_weight=0.01,
        ),
    )

    lqr_controller = LQRController(
        sample_rate=SW_SAMPLE_RATE,
    )

    bodyrate_controller = BodyrateController(
        y0 = np.zeros(4),
        sample_rate=SW_SAMPLE_RATE,
        params=BodyrateControllerParams(
            P = 25 * np.eye(3),
        ),
    )

    # Create simulator
    curve_planner.inputs_to(dfb_planner)
    dfb_planner.inputs_to(linearization)
    linearization.inputs_to(lqr_controller)
    lqr_controller.inputs_to(bodyrate_controller)
    bodyrate_controller.inputs_to(allocator)
    allocator.inputs_to(quadrotor)

    dfb_planner.feedback_from(quadrotor)

    linearization.feedback_from(quadrotor)
    linearization.feedback_from(bodyrate_controller)

    lqr_controller.feedback_from(quadrotor)
    lqr_controller.feedback_from(linearization)

    bodyrate_controller.feedback_from(quadrotor)
    bodyrate_controller.feedback_from(lqr_controller)

    model_graph = ModelGraph(
        models=[
            curve_planner,
            dfb_planner,
            linearization,
            lqr_controller,
            bodyrate_controller,
            allocator,
            quadrotor,
        ]
    )

    simulator = Simulator(model_graph, RK4(h=0.05))

    # Create visualization environment
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")

    ax.set_xlim(curve.center[0] - curve.radius - 1, curve.center[0] + curve.radius + 1)
    ax.set_ylim(curve.center[1] - curve.radius - 1, curve.center[1] + curve.radius + 1)
    ax.set_zlim(curve.center[2] - curve.radius - 1, curve.center[2] + curve.radius + 1)

    plot_env = (
        PlotEnvironment(fig, ax, t_range, frame_rate=20, t_start=0.0)
        .attach_element(
            QuadrotorFrameElement(
                system=quadrotor,
                color="black",
                alpha=1.0,
            )
        )
        .attach_element(
            TraveledPathElement(
                system=quadrotor,
            )
        )
        .attach_element(
            ParametricCurveElement(
                curve=curve
            )
        )
    )

    # Run simulation
    env = SimulationEnvironment(
        simulator=simulator,
        plot_env=plot_env,
    ).run_simulation(t_range=t_range, realtime=False, show_time=True)


if __name__ == "__main__":
    hover_stabilize()
    # circle_follow()
