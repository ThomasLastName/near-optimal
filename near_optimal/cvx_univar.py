
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Given data
k = 15
m = 2*k
x = np.linspace(-1,1,m)
y = np.cos(2*np.pi*x)/3

def empirical_risk_minimization( x, y, tol=None ):
    #
    # ~~~ Variables
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
        # constraints.append( x[2*j-1] * p[j-1] - x[2*j+1-1] * s[j-1] <= numerator )
        # constraints.append( numerator <= x[2*j+1-1] * p[j-1] - x[2*j-1] * s[j-1] )
    # #
    # # ~~~ Linear equality constraints for the dummy variables p and s
    # for j in range(k-1):
    #     s_j      = (z[2*j-1]   - z[2*j-1-1]) / (x[2*j-1]   - x[2*j-1-1])
    #     s_jplus1 = (z[2*j+2-1] - z[2*j+1-1]) / (x[2*j+2-1] - x[2*j+1-1])
    #     constraints.append( p[j-1] - s[j-1] == s_jplus1 - s[j-1] )
    #
    # ~~~ Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)
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
    # #
    # # ~~~
    # z_opt = z.value
    # p_opt = p.value
    # s_opt = s.value
    # nodes = []
    # for j in range(k-1):
    #     j += 1
    #     m_j      = (x[2*j-1]   + x[2*j-1-1]) / 2    # ~~~ (x_{2j}   + x_{2j-1})/2
    #     m_jplus1 = (x[2*j+2-1] + x[2*j+1-1]) / 2    # ~~~ (x_{2j+2} + x_{2j+1})/2
    #     a_j      = (z_opt[2*j-1]   + z_opt[2*j-1-1]) / 2    # ~~~ (z_{2j}   + z_{2j-1})/2
    #     a_jplus1 = (z_opt[2*j+2-1] + z_opt[2*j+1-1]) / 2    # ~~~ (z_{2j+2} + z_{2j+1})/2
    #     d_j      = x[2*j-1]   - x[2*j-1-1]          # ~~~ \delta_j     = x_{2j}   - x_{2j-1}
    #     d_jplus1 = x[2*j+2-1] - x[2*j+1-1]          # ~~~ \delta_{j+1} = x_{2j+2} - x_{2j+1}
    #     s_j      = (z_opt[2*j-1]   - z_opt[2*j-1-1]) / d_j
    #     s_jplus1 = (z_opt[2*j+2-1] - z_opt[2*j+1-1]) / d_jplus1
    #     # s_j      = (z_opt[2*j-1]   - z_opt[2*j-1-1]) / (x[2*j-1]   - x[2*j-1-1])
    #     # s_jplus1 = (z_opt[2*j+2-1] - z_opt[2*j+1-1]) / (x[2*j+2-1] - x[2*j+1-1])
    #     numerator   =   s_jplus1*m_jplus1 - s_j*m_j - (a_jplus1 - a_j)
    #     node = numerator/(s_jplus1-s_j)
    #     nodes.append(node)
    #     # assert abs( s_jplus1-s_j - (p_opt[j-1]-s_opt[j-1]) ) < tol
    #     # assert node>=x[2*j-1] and node<=x[2*j+1-1]  # ~~~ is between x_{2j} and x_{2j+1}
    # return nodes, z_opt


breakpoints,z_opt = empirical_risk_minimization(x,y)


for j in range(k-1):
    node = breakpoints[j]
    j += 1
    print( node-x[2*j-1] )
    print( x[2*j+1-1] - node )


"""
def empirical_risk_minimization( x, y, penalty_coefficient=100., tol=None ):
    #
    # ~~~ Variables
    z = cp.Variable(m)
    p = cp.Variable(k-1, nonneg=True)
    #
    # ~~~ Objective: minimize max_j |y_j - z_j|
    objective = cp.Minimize( cp.max(cp.abs(y-z)) + penalty_coefficient*cp.max(p) )
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
        d_j      = x[2*j-1]   - x[2*j-1-1]          # ~~~ \delta_j     = x_{2j}   - x_{2j-1}
        d_jplus1 = x[2*j+2-1] - x[2*j+1-1]          # ~~~ \delta_{j+1} = x_{2j+2} - x_{2j+1}
        a_j      = (z[2*j-1]   + z[2*j-1-1]) / 2    # ~~~ (z_{2j}   + z_{2j-1})/2
        a_jplus1 = (z[2*j+2-1] + z[2*j+1-1]) / 2    # ~~~ (z_{2j+2} + z_{2j+1})/2
        s_j      = (z[2*j-1]   - z[2*j-1-1]) / d_j
        s_jplus1 = (z[2*j+2-1] - z[2*j+1-1]) / d_jplus1
        c_j      = (x[2*j-1] + x[2*j+1-1]) / 2  # ~~~ midpoint (x_{2j} + x_{2j+1})/2 of the inverval where one break point is allowed
        D_j      = (x[2*j+1-1] - x[2*j-1])      # ~~~ length of the inverval where one break point is allowed
        numerator   =   s_jplus1*m_jplus1 - s_j*m_j - (a_jplus1 - a_j) - (s_jplus1-s_j)*c_j
        constraints.append( -D_j/2*p[j-1] <= numerator )
        constraints.append( numerator <= D_j/2*p[j-1] )
        constraints.append( s_jplus1 - s_j <= p[j-1] )
        constraints.append( -p[j-1] <= s_jplus1 - s_j )
    #
    # ~~~ Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)
    print(f"Optimal z: {z.value}")
    print(f"Optimal p: {p.value}")
    #
    # ~~~ Print debugging
    z = z.value
    p = p.value
    nodes = []
    for j in range(k-1):
        j += 1
        m_j      = (x[2*j-1]   + x[2*j-1-1]) / 2    # ~~~ (x_{2j}   + x_{2j-1})/2
        m_jplus1 = (x[2*j+2-1] + x[2*j+1-1]) / 2    # ~~~ (x_{2j+2} + x_{2j+1})/2
        d_j      = x[2*j-1]   - x[2*j-1-1]          # ~~~ \delta_j     = x_{2j}   - x_{2j-1}
        d_jplus1 = x[2*j+2-1] - x[2*j+1-1]          # ~~~ \delta_{j+1} = x_{2j+2} - x_{2j+1}
        a_j      = (z[2*j-1]   + z[2*j-1-1]) / 2    # ~~~ (z_{2j}   + z_{2j-1})/2
        a_jplus1 = (z[2*j+2-1] + z[2*j+1-1]) / 2    # ~~~ (z_{2j+2} + z_{2j+1})/2
        s_j      = (z[2*j-1]   - z[2*j-1-1]) / d_j
        s_jplus1 = (z[2*j+2-1] - z[2*j+1-1]) / d_jplus1
        node     = -(s_jplus1*m_jplus1 - s_j*m_j - (a_jplus1 - a_j)) / (s_jplus1-s_j)
        nodes.append(node)
        if tol is not None:
            assert node - x[2*j-1] > -tol   # ~~~ node >= x_{2j}
            assert x[2*j+1-1] - node > -tol # ~~~ node <= x_{2j+1}
    return nodes,z



# def empirical_risk_minimization( x, y, tol=None ):
#     #
#     # ~~~ Variables
#     z = cp.Variable(m)
#     #
#     # ~~~ Objective: minimize max_j |y_j - z_j|
#     objective = cp.Minimize( cp.max(cp.abs(y-z)) )
#     #
#     # ~~~ Constraints
#     constraints = []
#     #
#     # ~~~ Non-negativity constraints are already ensured by nonneg=True
#     pass
#     #
#     # ~~~ Linear inequality and equality constraints
#     for j in range(k-1):
#         j += 1
#         m_j      = (x[2*j-1]   + x[2*j-1-1]) / 2    # ~~~ (x_{2j}   + x_{2j-1})/2
#         m_jplus1 = (x[2*j+2-1] + x[2*j+1-1]) / 2    # ~~~ (x_{2j+2} + x_{2j+1})/2
#         d_j      = x[2*j-1]   - x[2*j-1-1]          # ~~~ \delta_j     = x_{2j}   - x_{2j-1}
#         d_jplus1 = x[2*j+2-1] - x[2*j+1-1]          # ~~~ \delta_{j+1} = x_{2j+2} - x_{2j+1}
#         a_j      = (z[2*j-1]   + z[2*j-1-1]) / 2    # ~~~ (z_{2j}   + z_{2j-1})/2
#         a_jplus1 = (z[2*j+2-1] + z[2*j+1-1]) / 2    # ~~~ (z_{2j+2} + z_{2j+1})/2
#         s_j      = (z[2*j-1]   - z[2*j-1-1]) / d_j
#         s_jplus1 = (z[2*j+2-1] - z[2*j+1-1]) / d_jplus1
#         c_j      = (x[2*j-1] + x[2*j+1-1]) / 2  # ~~~ midpoint (x_{2j} + x_{2j+1})/2 of the inverval where one break point is allowed
#         D_j      = (x[2*j+1-1] - x[2*j-1])      # ~~~ length of the inverval where one break point is allowed
#         numerator   =   s_jplus1*m_jplus1 - s_j*m_j - (a_jplus1 - a_j) + (s_jplus1-s_j)*c_j
#         constraints.append( -D_j/2*cp.abs(s_jplus1-s_j) <= numerator )
#         constraints.append( numerator <= D_j/2*cp.abs(s_jplus1-s_j) )
#     #
#     # ~~~ Solve
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver=cp.ECOS)
#     print(f"Optimal z: {z.value}")
#     #
#     # ~~~ Print debugging
#     z = z.value
#     p = p.value
#     nodes = []
#     for j in range(k-1):
#         j += 1
#         m_j      = (x[2*j-1]   + x[2*j-1-1]) / 2    # ~~~ (x_{2j}   + x_{2j-1})/2
#         m_jplus1 = (x[2*j+2-1] + x[2*j+1-1]) / 2    # ~~~ (x_{2j+2} + x_{2j+1})/2
#         d_j      = x[2*j-1]   - x[2*j-1-1]          # ~~~ \delta_j     = x_{2j}   - x_{2j-1}
#         d_jplus1 = x[2*j+2-1] - x[2*j+1-1]          # ~~~ \delta_{j+1} = x_{2j+2} - x_{2j+1}
#         a_j      = (z[2*j-1]   + z[2*j-1-1]) / 2    # ~~~ (z_{2j}   + z_{2j-1})/2
#         a_jplus1 = (z[2*j+2-1] + z[2*j+1-1]) / 2    # ~~~ (z_{2j+2} + z_{2j+1})/2
#         s_j      = (z[2*j-1]   - z[2*j-1-1]) / d_j
#         s_jplus1 = (z[2*j+2-1] - z[2*j+1-1]) / d_jplus1
#         node     = -(s_jplus1*m_jplus1 - s_j*m_j - (a_jplus1 - a_j)) / (s_jplus1-s_j)
#         nodes.append(node)
#         if tol is not None:
#             assert node - x[2*j-1] > -tol   # ~~~ node >= x_{2j}
#             assert x[2*j+1-1] - node > -tol # ~~~ node <= x_{2j+1}
#     return nodes,z
"""

