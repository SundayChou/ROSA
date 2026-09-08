import jax
import functools

from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein


@functools.partial(jax.jit, static_argnames=['epsilon', 'tau_a', 'tau_b', 'max_gw_iterations',
                                             'max_sinkhorn_iterations', 'gw_threshold', 'sinkhorn_threshold'])
def solve_unbalanced_gw(coords_source, coords_target, a=None, b=None, epsilon=2e-2, tau_a=0.999, tau_b=0.999, 
                        max_gw_iterations=100, max_sinkhorn_iterations=1000, gw_threshold=1e-2, sinkhorn_threshold=1e-2):
    geom_source = pointcloud.PointCloud(x=coords_source, epsilon=epsilon, scale_cost='mean')
    geom_target = pointcloud.PointCloud(x=coords_target, epsilon=epsilon, scale_cost='mean')

    prob = quadratic_problem.QuadraticProblem(
        geom_xx=geom_source,
        geom_yy=geom_target,
        a=a,
        b=b,
        tau_a=tau_a,
        tau_b=tau_b
    )

    inner_solver = sinkhorn.Sinkhorn(
        threshold=sinkhorn_threshold,
        max_iterations=max_sinkhorn_iterations
    )

    solver = gromov_wasserstein.GromovWasserstein(
        linear_solver=inner_solver,
        epsilon=epsilon, 
        threshold=gw_threshold,
        max_iterations=max_gw_iterations
    )

    out = solver(prob)
    pi = out.matrix

    return pi