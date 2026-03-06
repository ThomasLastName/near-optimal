"""
Simon asked to try this:
Don't just stop at uniform weights. Update them and try again.
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
from quality_of_life.my_plt_utils import points_with_curves
from quality_of_life.my_cvx_utils import solve_dual_of_QCQP
from quadratic_univar import DualSpline, x_train, y_train, x_test, y_test, f


v = DualSpline( x_train, y_train )
b_star = v.solve_dual_of_mse_minimization(weighted_mean=True)


def plot():
    with torch.no_grad(): pred = v(x_train)
    max_abs_error = (pred-y_train).abs().max().item()
    mean_sq_error = ((pred-y_train)**2).mean().item()
    suboptimality_ratio = max_abs_error/b_star
    fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), show=False, title=f"The Result of Some Quadratic Program (sub-optimality ratio: {suboptimality_ratio:.4f})" )
    with torch.no_grad():
        nodes = v.compute_break_points()
        ax.scatter( nodes, v(nodes), color="blue", alpha=0.4 )
    plt.show()


def minimize_weighted_mse( w:np.ndarray=None, v:DualSpline=v ):
    #
    # ~~~ Build and solve problem
    aat_minus_bbt = -v.bbt_minus_aat.cpu().numpy()
    y = v.y.cpu().numpy()
    m = len(y)
    if w is None: w = np.ones(m)/m
    H_o =  np.diag(w)
    c_o = -w*y
    d_o =  (1/m)*(w*y**2).sum()
    problem, _, z = solve_dual_of_QCQP(
            H_o,
            c_o,
            d_o,
            H_I = aat_minus_bbt,
            c_I = (v.k-1)*[np.zeros(m)],
            d_I = (v.k-1)*[0], 
            solver     = "SCS",
            eps_abs    = 1e-4,
            eps_rel    = 1e-4,
            eps_infeas = 1e-7
        )
    #
    # ~~~ Process results
    v.z.data = torch.from_numpy(z)
    v.problem = problem
    v.project()
    v.update_upper_bound_on_primal()
    dual_max = np.sqrt(problem.objective.value)
    v.lower_bound_on_primal = max( v.lower_bound_on_primal, dual_max )
    return z, w, dual_max


w = None
y = y_train.numpy()
for j in range(5):
    z, w, d = minimize_weighted_mse(w)
    plot()
    print("")
    print(f"    Error {max(errors)} after {j} iters")
    print("")
    errors = abs(z-y)
    w = (errors + 1e-3)/sum(errors + 1e-3)


