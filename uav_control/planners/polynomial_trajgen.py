import math
from typing import Tuple

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar


def _qp_var_indices(segment_index, poly_order, m=0):
    return segment_index * (poly_order + 1), (segment_index + 1) * (poly_order + 1) - m


def _qp_constraints(waypoints, times, order=7, n_constrained_end_derivs=2):
    N = order
    M = len(waypoints) - 1
    t0, tf = times[0], times[-1]

    n_constraints_endpoint_matches = 2 * M
    n_constraints_deriv_continuity = M - 1
    n_constraints_start_from_rest = 1
    n_constraints_end_at_rest = 4

    n_constraints = (
        n_constraints_endpoint_matches
        + n_constraints_deriv_continuity
        + n_constraints_start_from_rest
        + n_constraints_end_at_rest
    )

    polyderivs = [np.polyder(np.ones(N + 1), m=i) for i in range(0, N + 1)]

    A = np.zeros(shape=(n_constraints, (N + 1) * M))
    b = np.zeros(shape=(n_constraints,))
    row_index = 0

    # Constraint type 1: the polynomial must pass through the waypoints at the specified times
    for i in range(M):
        seg_t0, seg_tf = times[i], times[i + 1]
        seg_w0, seg_wf = waypoints[i], waypoints[i + 1]

        A_col_idx0, A_col_idxf = _qp_var_indices(i, N, m=0)

        t0_coeffs = seg_t0 ** np.arange(N, -1, -1)
        tf_coeffs = seg_tf ** np.arange(N, -1, -1)

        A[row_index, A_col_idx0:A_col_idxf] = t0_coeffs
        b[row_index] = seg_w0
        row_index += 1

        A[row_index, A_col_idx0:A_col_idxf] = tf_coeffs
        b[row_index] = seg_wf
        row_index += 1

    # Constraint type 2: the first derivative of the polynomial must be continuous at the segment boundaries
    for i in range(M - 1):
        seg_curr_tf, seg_next_t0 = times[i + 1], times[i + 1]

        A_seg_curr_col_idx0, A_seg_curr_col_idxf = _qp_var_indices(i, N, m=1)
        A_seg_next_col_idx0, A_seg_next_col_idxf = _qp_var_indices(i + 1, N, m=1)

        poly_deriv_1_coeffs = polyderivs[1]
        seg_curr_tf_deriv_coeffs = poly_deriv_1_coeffs * (seg_curr_tf ** np.arange(N - 1, -1, -1))
        seg_next_t0_deriv_coeffs = poly_deriv_1_coeffs * (seg_next_t0 ** np.arange(N - 1, -1, -1))

        A[row_index, A_seg_curr_col_idx0:A_seg_curr_col_idxf] = seg_curr_tf_deriv_coeffs
        A[row_index, A_seg_next_col_idx0:A_seg_next_col_idxf] = -seg_next_t0_deriv_coeffs
        row_index += 1

    # Constraint type 3: first and last segment start/end must have zero-value derivatives up to order `n_constrained_end_derivs`
    # Constraint type 3: the first segment starts from rest (v0=0)
    seg_index = 0
    A_curr_col_idx0, A_curr_col_idxf = _qp_var_indices(seg_index, N, m=1)

    poly_deriv_order_coeffs = polyderivs[1]
    t_deriv_coeffs = poly_deriv_order_coeffs * (t0 ** np.arange(N - 1, -1, -1))

    A[row_index, A_curr_col_idx0:A_curr_col_idxf] = t_deriv_coeffs
    row_index += 1

    # Constraint type 4: the last segment ends at rest (all derivatives up to 4th order=0)
    seg_index = M - 1
    for deriv_order in range(1, 4 + 1):
        A_curr_col_idx0, A_curr_col_idxf = _qp_var_indices(seg_index, N, m=deriv_order)

        poly_deriv_order_coeffs = polyderivs[deriv_order]
        t_deriv_coeffs = poly_deriv_order_coeffs * (tf ** np.arange(N - deriv_order, -1, -1))

        A[row_index, A_curr_col_idx0:A_curr_col_idxf] = t_deriv_coeffs
        row_index += 1

    # for i, t in enumerate([t0, tf]):
    #     seg_index = 0 if i == 0 else M - 1

    #     for deriv_order in range(1, n_constrained_end_derivs + 1):
    #         A_curr_col_idx0, A_curr_col_idxf = _qp_var_indices(seg_index, N, m=deriv_order)

    #         poly_deriv_order_coeffs = polyderivs[deriv_order]
    #         t_deriv_coeffs = poly_deriv_order_coeffs * (t ** np.arange(N - deriv_order, -1, -1))

    #         A[row_index, A_curr_col_idx0 : A_curr_col_idxf] = t_deriv_coeffs
    #         row_index += 1

    return A, b


def _qp_objective_quad_form(waypoints, times, order=7, minimize_order=1):
    N = order
    M = len(waypoints) - 1
    p = minimize_order

    H_array = [np.zeros(shape=(N + 1, N + 1), dtype=np.float64) for _ in range(M)]
    factorials = [math.factorial(i) for i in range(N + 1)]

    for k in range(M):
        H = H_array[k]
        seg_t0, seg_tf = times[k], times[k + 1]

        for i in range(p, N + 1):
            for j in range(p, N + 1):
                new_i = N - i
                new_j = N - j

                exponent = i + j - 2 * p + 1

                H[new_i, new_j] = (
                    factorials[i]
                    * factorials[j]
                    / (factorials[i - p] * factorials[j - p])
                    * 1
                    / exponent
                    * (seg_tf**exponent - seg_t0**exponent)
                )

    # Final H matrix is block diagonal, one matrix for each segment
    return block_diag(*H_array)


def traj_opt_1d(
    waypoints, times, order=7, n_constrained_end_derivs=2, minimize_order=None
) -> np.ndarray:
    assert len(waypoints) == len(times)

    N = order
    M = len(waypoints) - 1

    # Get constraint matrix A, vector b s.t. A * x = b
    A, b = _qp_constraints(
        waypoints=waypoints,
        times=times,
        order=order,
        n_constrained_end_derivs=n_constrained_end_derivs,
    )

    if minimize_order is None:
        x, _residual, _rank, _singular_values = np.linalg.lstsq(A, b, rcond=None)
        return x

    # Get objective function constant H s.t. we minimize x^T * H * x
    H = _qp_objective_quad_form(
        waypoints=waypoints, times=times, order=order, minimize_order=minimize_order
    )

    x = cp.Variable((N + 1) * M)

    # assume_PSD=True, since H is taken from coefficients of x, evaluations at t>=0
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, H, assume_PSD=True)), [A @ x == b])
    prob.solve(solver="SCS")

    return x.value


class PolynomialTrajectoryND:
    def __init__(
        self,
        waypoints,
        timepoints,
        order=7,
        n_constrained_end_derivs=2,
        minimize_order=None,
    ):
        assert minimize_order is None or minimize_order <= order

        self.dim = waypoints.shape[1]
        self.waypoints = waypoints  # (N, dim) array of position waypoints
        self.timepoints = timepoints  # (N,) array of floats
        self.order = order
        self.n_constrained_end_derivs = n_constrained_end_derivs
        self.minimize_order = minimize_order

        self.n_segments = len(waypoints) - 1
        self.t0, self.tf = timepoints[0], timepoints[-1]
        self.w0, self.wf = waypoints[0], waypoints[-1]

        self.polyderivs = [
            np.polyder(np.ones(self.order + 1), m=i) for i in range(0, self.order + 1)
        ]

        # List of coeffs s.t. polynomial_coeffs[i] are coeffs for i^th dimension of waypoint
        self.polynomial_coeffs = self._generate_polynomial_coeffs()

    def closest_waypoint_t(self, r_d: np.ndarray, bounds: Tuple[float, float]) -> float:
        assert r_d.size == self.dim

        def objective_fn(t) -> Tuple[float, float]:
            [r, v] = self(t, n_derivatives=1)

            r_error = r_d - r
            objective_value = np.linalg.norm(r_error)

            return objective_value

        minimize_result = minimize_scalar(fun=objective_fn, method="bounded", bounds=bounds)
        if minimize_result.success:
            return minimize_result.x
        else:
            raise RuntimeError("Optimization failed to find closest waypoint time")

    def _generate_polynomial_coeffs(self):
        return np.array(
            [
                traj_opt_1d(
                    waypoints=self.waypoints[:, i],
                    times=self.timepoints,
                    order=self.order,
                    n_constrained_end_derivs=self.n_constrained_end_derivs,
                    minimize_order=self.minimize_order,
                )
                for i in range(self.dim)
            ]
        )  # Shape (dim, (order + 1) * n_segments)

    def __call__(self, t, n_derivatives=0):
        assert n_derivatives >= 0

        if t > self.tf:
            t = self.tf
        elif t < self.t0:
            t = self.t0

        segment_index = np.searchsorted(self.timepoints, t) - 1
        segment_index = min(self.n_segments - 1, max(0, segment_index))

        t_powers = t ** np.arange(self.order, -1, -1)

        ret = [np.empty(self.dim) for _ in range(n_derivatives + 1)]

        for i in range(n_derivatives + 1):
            coeffs_idx0, coeffs_idxf = _qp_var_indices(segment_index, self.order, m=i)
            coeffs = self.polynomial_coeffs[:, coeffs_idx0:coeffs_idxf] * self.polyderivs[i]

            ret[i] = coeffs @ t_powers[i:]

        return ret


def visualize_spatial_trajectory(traj: PolynomialTrajectoryND):
    import matplotlib.pyplot as plt

    if traj.dim == 1:
        raise ValueError("1D trajectory cannot be spatially visualized")
    elif traj.dim == 2:
        ts = np.linspace(traj.t0, traj.tf, traj.n_segments * 120)

        xys = np.array([traj(t)[0] for t in ts])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.plot(
            xys[:, 0],
            xys[:, 1],
            linestyle="--",
            linewidth=1,
            color="gray",
            label="Path",
        )  # 'o' adds markers for each point
        ax.scatter(
            traj.waypoints[:, 0],
            traj.waypoints[:, 1],
            marker="o",
            color="red",
            label="Waypoints",
        )

        for i in range(len(traj.waypoints)):
            ax.text(
                traj.waypoints[i, 0] + 0.1,
                traj.waypoints[i, 1] + 0.1,
                f"{i}",
                fontsize=12,
                color="red",
                fontfamily="monospace",
                va="center",
                ha="center",
            )

        ax.legend()
        ax.set_title("2D Spatial Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        return fig
    elif traj.dim == 3:
        ts = np.linspace(traj.t0, traj.tf, traj.n_segments * 120)

        xyzs = np.array([traj(t)[0] for t in ts])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            xyzs[:, 0],
            xyzs[:, 1],
            xyzs[:, 2],
            linestyle="--",
            linewidth=1,
            color="gray",
            label="Path",
        )
        ax.scatter(
            traj.waypoints[:, 0],
            traj.waypoints[:, 1],
            traj.waypoints[:, 2],
            color="red",
            label="Waypoints",
        )

        for i in range(len(traj.waypoints)):
            ax.text(
                traj.waypoints[i, 0] + 0.1,
                traj.waypoints[i, 1] + 0.1,
                traj.waypoints[i, 2] + 0.1,
                f"{i}",
                fontsize=12,
                color="red",
                fontfamily="monospace",
                va="center",
                ha="center",
            )

        ax.legend()
        ax.set_title("3D Spatial Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        return fig

    raise ValueError("Cannot visualize trajectories of dimension > 3")
