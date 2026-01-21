
import cvxpy as cp
import torch
import matplotlib.pyplot as plt
from quality_of_life.my_plt_utils import points_with_curves
from near_optimal.gradient_univar import spline
from near_optimal.quadratic_univar import x_train, y_train, f, DualSpline


def empirical_risk_minimization( x, y, tol=None ):
    #
    # ~~~ Variables
    m = len(x)
    k = int(m/2)
    z = cp.Variable(m)
    p = cp.Variable(k-1, nonneg=True)
    s = cp.Variable(k-1, nonneg=True)
    #
    # ~~~ Objective: minimize max_j |y_j - z_j|
    objective = cp.Minimize( cp.max(cp.abs(y-z)) )
    #
    # ~~~ Constraints
    constraints = []
    #
    # ~~~ Non-negativity constraints are already ensured by nonneg=True
    pass
    #
    # ~~~ Linear inequality and equality constraints
    for j in range(k-1):
        j += 1
        m_j      = (x[2*j-1]   + x[2*j-1-1]) / 2    # ~~~ (x_{2j}   + x_{2j-1})/2
        m_jplus1 = (x[2*j+2-1] + x[2*j+1-1]) / 2    # ~~~ (x_{2j+2} + x_{2j+1})/2
        d_j      = x[2*j-1]    - x[2*j-1-1]         # ~~~ \delta_j     = x_{2j}   - x_{2j-1}
        d_jplus1 = x[2*j+2-1]  - x[2*j+1-1]         # ~~~ \delta_{j+1} = x_{2j+2} - x_{2j+1}
        a_j      = (z[2*j-1]   + z[2*j-1-1]) / 2    # ~~~ (z_{2j}   + z_{2j-1})/2
        a_jplus1 = (z[2*j+2-1] + z[2*j+1-1]) / 2    # ~~~ (z_{2j+2} + z_{2j+1})/2
        s_j      = (z[2*j-1]   - z[2*j-1-1]) / d_j
        s_jplus1 = (z[2*j+2-1] - z[2*j+1-1]) / d_jplus1
        c_j      = (x[2*j-1] + x[2*j+1-1]) / 2  # ~~~ midpoint (x_{2j} + x_{2j+1})/2 of the inverval where one break point is allowed
        D_j      = (x[2*j+1-1] - x[2*j-1])      # ~~~ length of the inverval where one break point is allowed
        numerator   =   s_jplus1*m_jplus1 - s_j*m_j - (a_jplus1 - a_j) - (s_jplus1-s_j)*c_j
        constraints.append( -D_j/2*(p[j-1]+s[j-1]) <= numerator )
        constraints.append( numerator <= D_j/2*(p[j-1]+s[j-1]) )
        constraints.append( p[j-1] - s[j-1] == s_jplus1 - s_j )
    #
    # ~~~ Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    print(f"Optimal z: {z.value}")
    print(f"Optimal p: {p.value}")
    print(f"Optimal s: {s.value}")
    #
    # ~~~ Print debugging
    z = z.value
    p = p.value
    s = s.value
    nodes = []
    for j in range(k-1):
        j += 1
        m_j      = (x[2*j-1]   + x[2*j-1-1]) / 2    # ~~~ (x_{2j}   + x_{2j-1})/2
        m_jplus1 = (x[2*j+2-1] + x[2*j+1-1]) / 2    # ~~~ (x_{2j+2} + x_{2j+1})/2
        d_j      = x[2*j-1]    - x[2*j-1-1]         # ~~~ \delta_j     = x_{2j}   - x_{2j-1}
        d_jplus1 = x[2*j+2-1]  - x[2*j+1-1]         # ~~~ \delta_{j+1} = x_{2j+2} - x_{2j+1}
        a_j      = (z[2*j-1]   + z[2*j-1-1]) / 2    # ~~~ (z_{2j}   + z_{2j-1})/2
        a_jplus1 = (z[2*j+2-1] + z[2*j+1-1]) / 2    # ~~~ (z_{2j+2} + z_{2j+1})/2
        s_j      = (z[2*j-1]   - z[2*j-1-1]) / d_j
        s_jplus1 = (z[2*j+2-1] - z[2*j+1-1]) / d_jplus1
        node     = (s_jplus1*m_jplus1 - s_j*m_j - (a_jplus1 - a_j)) / (s_jplus1-s_j)
        nodes.append(node)
        if tol is not None:
            assert abs( s_jplus1-s_j - (p[j-1]-s[j-1]) ) < tol
            assert node - x[2*j-1] > -tol   # ~~~ node - x_{2j} >= 0
            assert x[2*j+1-1] - node > -tol # ~~~ x_{2j+1} - node >= 0
    return nodes, z


if __name__ == "__main__":
    #
    # ~~~ Get some data
    k = 15
    m = 2*k
    x = x_train.numpy()
    y = y_train.numpy()
    #
    # ~~~ Test
    breakpoints, z_opt = empirical_risk_minimization(x,y)
    print("")
    print(f"This minimization problem is too relaxed! The objective value of the relaxed problem is zero (technically, {min(abs(z_opt-y))}) when we know from the other experiments that it be > 0.048.")
    print("")
    # for j in range(k-1):
    #     node = breakpoints[j]
    #     j += 1
    #     print( node-x[2*j-1] )
    #     print( x[2*j+1-1] - node )
    # v = spline(x,z_opt)
    # fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), title="Too Relaxed... Objective is Zero When it Shouldn't Be", show=False )
    # with torch.no_grad():
    #     nodes = v.compute_break_points()
    #     ax.scatter( nodes, v(nodes), color="blue", alpha=0.4 )
    # plt.show()