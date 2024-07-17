import numpy as np
import spatialmath as sm
import matplotlib.pyplot as plt
import os

from hybrid_ode_sim.utils.logging_tools import LogLevel
from hybrid_ode_sim.simulation.simulator import Simulator, ModelGraph
from hybrid_ode_sim.simulation.ode_solvers.fixed_step_solver import RK4
from hybrid_ode_sim.simulation.ode_solvers.adaptive_step_solver import RK23, RK45
from hybrid_ode_sim.simulation.rendering.base import PlotEnvironment

from uav_control.utils.find_package_root import find_package_root
from uav_control.constants import g, compose_state, R_B0_N, Q_NB, V_B0_N, OMEGA_B0_B
from uav_control.dynamics import QuadrotorRigidBodyParams, QuadrotorRigidBodyDynamics
from uav_control.controllers.geometric_controller import QuadrotorGeometricControllerParams, QuadrotorGeometricController
from uav_control.planners.waypoint_planner import QuadrotorWaypointPlannerParams, QuadrotorWaypointPlanner
from uav_control.planners.point_stabilize_planner import QuadrotorStabilizationPlanner, QuadrotorStabilizationPlannerParams
from uav_control.planners.polynomial_planner import QuadrotorPolynomialPlanner, QuadrotorPolynomialPlannerParams
from uav_control.allocators.qp_control_allocator import QuadrotorQPAllocatorParams, QuadrotorQPAllocator

from uav_control.rendering.polynomial_path import PolynomialTrajectory
from uav_control.rendering.coordinate_frame import CoordinateFrame
from uav_control.rendering.waypoint_path import WaypointTrajectory
from uav_control.rendering.quadrotor_frame import QuadrotorFrame

np.set_printoptions(precision=3, suppress=True)

SW_SAMPLE_RATE = 50  # Hz
M = 4.34
I = np.diag([0.0820, 0.0845, 0.1377])
L = 0.315
D_drag = 0.0 * np.diag([0.26, 0.28, 0.42])
cq_ct_ratio = 8.004e-4

SAVED_FIGURES_PATH = os.path.join(find_package_root(__file__), 'docs')

def waypoint_test(save=False):
    waypoints = np.array([
        [0.0, 3.0, 2.0],
        [3.0, 3.0, 3.0],
        [3.0, 0.0, 2.0],
        [0.0, 0.0, 1.0],
    ])
    timepoints = 4.0 * np.arange(1, len(waypoints) + 1)
    
    planner = QuadrotorWaypointPlanner(
        y0=0,
        sample_rate=SW_SAMPLE_RATE,
        params=QuadrotorWaypointPlannerParams(
            waypoint_positions=waypoints,
            waypoint_times=timepoints,
        ),
    )

    rigid_body_params = QuadrotorRigidBodyParams(
        m=M,
        I=I,
        D_drag=D_drag
    )

    quadrotor = QuadrotorRigidBodyDynamics(
        y0=compose_state(
            r_b0_N=np.array([0.0, 0.0, 1.0]),
            q_NB=np.array([1.0, 0.0, 0.0, 0.0]),
            v_b0_N=np.array([0.0, 0.0, 0.0]),
            omega_b0_B=np.array([0.0, 0.0, 0.0])
        ),
        params=rigid_body_params,
        logging_level=LogLevel.ERROR
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
            cq_ct_ratio=cq_ct_ratio
        )
    )
    
    controller = QuadrotorGeometricController(
        y0=np.array([1.0, 0.0, 0.0, 0.0]),
        sample_rate=SW_SAMPLE_RATE,
        params=QuadrotorGeometricControllerParams(
            kP=6 * np.diag([1.0, 1.0, 1.0]),
            kD=8 * np.diag([1.0, 1.0, 1.0]),
            kR=8.81 * np.diag([1.0, 1.0, 1.0]),
            kOmega=2.54 * np.diag([1.0, 1.0, 1.0]),
            m=rigid_body_params.m,
            I=rigid_body_params.I,
            # D_drag=D_drag,
        )
    )
    
    # Create simulator
    planner.inputs_to(controller)
    controller.inputs_to(allocator)
    allocator.inputs_to(quadrotor)
    
    planner.feedback_from(quadrotor)
    controller.feedback_from(quadrotor)
    
    model_graph = ModelGraph(models=[controller, quadrotor, allocator, planner])
    
    t_range=[0, timepoints[-1] + 2.0]
    simulator = Simulator(model_graph, RK4(h=0.01))
    simulator.simulate(t_range)
    
    # Create visualization
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_zlim(-1, 4)
    
    env = PlotEnvironment(fig, ax, t_range, frame_rate=20, t_start=0.0)
    quadrotor_frame = QuadrotorFrame(env, system=quadrotor)
    path = WaypointTrajectory(env, waypoints, waypoint_color='red')

    env.render(plot_elements=[quadrotor_frame, path], show_time=True,
               save=save, save_path=f'{SAVED_FIGURES_PATH}/geometric_control/waypoint_test.mp4')

def upside_down_test(save=False):
    planner = QuadrotorStabilizationPlanner(
        y0=0,
        sample_rate=SW_SAMPLE_RATE,
        params=QuadrotorStabilizationPlannerParams(
            position=np.array([0.0, 0.0, 1.0]),
            b1d=np.array([1.0, 0.0, 0.0])
        ),
    )

    rigid_body_params = QuadrotorRigidBodyParams(
        m=M,
        I=I,
        D_drag=D_drag
    )

    quadrotor = QuadrotorRigidBodyDynamics(
        y0=compose_state(
            r_b0_N=np.array([0.0, 0.0, 1.0]),
            q_NB=sm.base.r2q(sm.base.rotx(178, 'deg')),
            v_b0_N=np.array([0.0, 0.0, 0.0]),
            omega_b0_B=np.array([0.0, 0.0, 0.0])
        ),
        params=rigid_body_params,
        logging_level=LogLevel.ERROR
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
            cq_ct_ratio=cq_ct_ratio
        )
    )
    
    controller = QuadrotorGeometricController(
        y0=np.array([1.0, 0.0, 0.0, 0.0]),
        sample_rate=SW_SAMPLE_RATE,
        params=QuadrotorGeometricControllerParams(
            kP=6 * np.diag([1.0, 1.0, 1.0]),
            kD=8 * np.diag([1.0, 1.0, 1.0]),
            kR=8.81 * np.diag([1.0, 1.0, 1.0]),
            kOmega=2.54 * np.diag([1.0, 1.0, 1.0]),
            m=rigid_body_params.m,
            I=rigid_body_params.I,
            # D_drag=D_drag,
        )
    )
    
    # Create simulator
    planner.inputs_to(controller)
    controller.inputs_to(allocator)
    allocator.inputs_to(quadrotor)
    
    planner.feedback_from(quadrotor)
    controller.feedback_from(quadrotor)
    
    model_graph = ModelGraph(models=[controller, quadrotor, allocator, planner])
    
    t_range=[0, 6.0]
    simulator = Simulator(model_graph, RK4(h=0.01))
    simulator.simulate(t_range)
    
    # Create visualization
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    env = PlotEnvironment(fig, ax, t_range, frame_rate=20, t_start=0.0)
    quadrotor_frame = QuadrotorFrame(env, system=quadrotor)

    env.render(plot_elements=[quadrotor_frame], show_time=True,
               save=save, save_path=f'{SAVED_FIGURES_PATH}/geometric_control/upside_down_test.mp4')

def trajectory_test(save=False):
    waypoints = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 3.0, 2.0],
        [3.0, 3.0, 3.0],
        [3.0, 0.0, 2.0],
        [0.0, 0.0, 1.0],
    ])
    timepoints = 2.0 * np.arange(0, len(waypoints), 1)
    
    planner = QuadrotorPolynomialPlanner(
        y0=0,
        sample_rate=SW_SAMPLE_RATE,
        params=QuadrotorWaypointPlannerParams(
            waypoint_positions=waypoints,
            waypoint_times=timepoints,
        ),
    )

    rigid_body_params = QuadrotorRigidBodyParams(
        m=M,
        I=I,
        D_drag=D_drag
    )

    quadrotor = QuadrotorRigidBodyDynamics(
        y0=compose_state(
            r_b0_N=np.array([0.0, 0.0, 1.0]),
            q_NB=np.array([1.0, 0.0, 0.0, 0.0]),
            v_b0_N=np.array([0.0, 0.0, 0.0]),
            omega_b0_B=np.array([0.0, 0.0, 0.0])
        ),
        params=rigid_body_params,
        logging_level=LogLevel.ERROR
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
            cq_ct_ratio=cq_ct_ratio
        )
    )
    
    controller = QuadrotorGeometricController(
        y0=np.array([1.0, 0.0, 0.0, 0.0]),
        sample_rate=SW_SAMPLE_RATE,
        params=QuadrotorGeometricControllerParams(
            kP=6 * np.diag([1.0, 1.0, 1.0]),
            kD=8 * np.diag([1.0, 1.0, 1.0]),
            kR=8.81 * np.diag([1.0, 1.0, 1.0]),
            kOmega=2.54 * np.diag([1.0, 1.0, 1.0]),
            m=rigid_body_params.m,
            I=rigid_body_params.I,
            # D_drag=D_drag,
        )
    )
    
    # Create simulator
    planner.inputs_to(controller)
    controller.inputs_to(allocator)
    allocator.inputs_to(quadrotor)
    
    planner.feedback_from(quadrotor)
    controller.feedback_from(quadrotor)
    
    model_graph = ModelGraph(models=[controller, quadrotor, allocator, planner])
        
    t_range=[0, timepoints[-1] + 2.0]
    simulator = Simulator(model_graph, RK4(h=0.01))
    simulator.simulate(t_range)
    
    # Create visualization
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_zlim(-1, 4)
    
    env = PlotEnvironment(fig, ax, t_range, frame_rate=20, t_start=0.0)
    quadrotor_frame = QuadrotorFrame(env, system=quadrotor)
    path = PolynomialTrajectory(env, planner.polynomial_traj)

    env.render(plot_elements=[quadrotor_frame, path], show_time=True,
               save=save, save_path=f'{SAVED_FIGURES_PATH}/geometric_control/polynomial_test.mp4')

if __name__ == "__main__":
    waypoint_test(save=False)
    upside_down_test(save=False)
    trajectory_test(save=False)