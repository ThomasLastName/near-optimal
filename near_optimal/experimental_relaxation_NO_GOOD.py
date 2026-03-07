
"""
This file tests and idea I tried for dealing with the constraint, but the idea doesn't work.
The constraint can be phrased as |s_{j+1}m_{j+1} - s_jm_j - (a_{j+1}-a_j) - (s_{j+1}-s_j)c_j| \leq D_j/2|s_{j+1}-s_j|.
A trick I've seen in linear program is to write |x| = a+b where a and b are dummy variables satisfying a,b \geq 0, x=a-b.
The idea is that a can represent the positive part of x, and b can represent the negative part.
I've seen it done before where this trick is used to convert a minimization problem involving |...| into a linear program.
So, I thought I'd try using p_j+n_j as a stand-in for |s_{j+1}-s_j|, where I intorduce non-negative vectors p and n
("p" for "positive" and "n" for "negative") satisfying the constraint s_{j+1}-s_j = p_j - n_j.
This doesn't work.
The relaxation |x| = a+b only yields an equivalent minimization problem if you're doing something like *minimizing* the absolute value
as that forces |x| = a+b to be as small as possible which, subject to the constraints, occurs when a=x^+ and b=x^-.
In the case at hand, the constraing |s_{j+1}m_{j+1} - s_jm_j - (a_{j+1}-a_j) - (s_{j+1}-s_j)c_j| \leq D_j/2|s_{j+1}-s_j|
wants to make the absolute value |s_{j+1}-s_j| big, not small.
Hence, the relaxed problem where |s_{j+1}-s_j| is replaced by p_j+n_j is too slack.
The upper bound p_j + n_j can be made arbitrarily large subject to the constraints p_j,n_j \geq 0 and s_{j+1}-s_j = p_j - n_j.
Hence, the upper bound doesn't end up applying, and the relaxed minimization problem simply chooses z==y. No good.
"""

import cvxpy as cp
import torch
import matplotlib.pyplot as plt
from quality_of_life.my_plt_utils import points_with_curves
from near_optimal.penalty_functions_NOT_RECOMMENDED import spline
from near_optimal.quadratic_univar import x_train, y_train, f, DualSpline


def linear_relaxation( x, y, tol=None ):
    #
    # ~~~ Variables
    m = len(x)
    k = int(m/2)
    z = cp.Variable(m)
    p = cp.Variable(k-1, nonneg=True)
    n = cp.Variable(k-1, nonneg=True)
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
        c_j      = (x[2*j-1] + x[2*j+1-1]) / 2      # ~~~ midpoint (x_{2j} + x_{2j+1})/2 of the inverval where one break point is allowed
        D_j      = (x[2*j+1-1] - x[2*j-1])          # ~~~ length of the inverval where one break point is allowed
        numerator   =   s_jplus1*m_jplus1 - s_j*m_j - (a_jplus1 - a_j) - (s_jplus1-s_j)*c_j
        constraints.append( -D_j/2*(p[j-1]+n[j-1]) <= numerator )
        constraints.append( numerator <= D_j/2*(p[j-1]+n[j-1]) )
        constraints.append( p[j-1] - n[j-1] == s_jplus1 - s_j )
    #
    # ~~~ Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    # print(f"Optimal z: {z.value}")
    # print(f"Optimal p: {p.value}")
    # print(f"Optimal n: {n.value}")
    #
    # ~~~ Print debugging
    z = z.value
    p = p.value
    n = n.value
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
            assert abs( s_jplus1-s_j - (p[j-1]-n[j-1]) ) < tol
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
    breakpoints, z_opt = linear_relaxation(x,y)
    print("")
    print(f"This minimization problem is too relaxed! The objective value of the relaxed problem is zero (technically, {min(abs(z_opt-y))}) when we know from the other experiments that it should be > 0.048.")
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
