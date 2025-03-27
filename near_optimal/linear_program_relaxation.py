
import numpy as np
import cvxpy as cvx
from matplotlib import pyplot as plt

np.random.seed(2025)
k = 15
m = 2*k
x_train = np.linspace(-1,1,m)
f = lambda x: np.sin(2*np.pi*x)
noise_level = 0.1
y_train = f(x_train) + noise_level*np.random.randn(m) + 2
x_train = np.concatenate([ [x_train.min()-1], x_train, [x_train.max()+1] ])

# def linear_relaxation(x,y):
if __name__ == "__main__":
    x = x_train
    y = y_train
    #
    # ~~~ Define all the variables
    eps = cvx.Variable(1)
    h   = cvx.Variable(k+1,nonneg=True)
    tau = cvx.Variable(k+1)
    T   = cvx.Variable(k+1)
    theta   = cvx.Variable(k)
    nu      = cvx.Variable(k)
    #
    # ~~~ Constrain the "epigraph variable" to be non-negative (optional) since the true objective is non-negative
    constraints = [eps>=0]
    #
    # ~~~ Constrain on the location of knots
    for j in range(k+1):
        constraints.append( x[2*j] <= tau[j] )
        constraints.append( tau[j] <= x[2*j+1] )
        # if j==0:    constraints.append( tau[j] <= x[2*j+1] )
        # elif j<k:   constraints.append( x[2*j] <= tau[j] <= x[2*j+1] )
        # else:       constraints.append( x[2*j] <= tau[j] )
    #
    # ~~~ Relaxation of the constraint T_j = \eps\tau_j (multiply the the constraint on \tau_j by \eps)
    for j in range(k+1):
        constraints.append( eps*x[2*j] <= T[j] )
        constraints.append( T[j] <= eps*x[2*j+1] )
        # if j==0:    constraints.append( T[j] <= eps*x[2*j+1] )
        # elif j<k:   constraints.append( eps*x[2*j] <= T[j] <= eps*x[2*j+1] )
        # else:       constraints.append( eps*x[2*j] <= T[j] )
    #
    # ~~~ Relaxation of the constraint \theta_j = h_{j-1}\tau_j (multiply the the constraint on \tau_j by h_{j-1})
    for j in range(k):
        j += 1
        constraints.append( h[j-1]*x[2*j] <= theta[j-1] )
        constraints.append( theta[j-1] <= h[j-1]*x[2*j+1] )
        # constraints.append( h[j]*x[2*j] <= theta[j] )
        # constraints.append( theta[j] <= h[j]*x[2*j+1] )
        # if j==0:    pass
        # elif j<k:   constraints.append( eps*x[2*j] <= T[j] <= eps*x[2*j+1] )
        # else:       constraints.append( eps*x[2*j] <= T[j] )
    #
    # ~~~ Relaxation of the constraint \nu_j = h_j\tau_{j-1} (multiply the the constraint on \tau_j by h_j)
    for j in range(k):
        constraints.append( h[j+1]*x[2*j] <= nu[j] )
        constraints.append( nu[j] <= h[j+1]*x[2*j+1] )
    #
    # ~~~ The epigraph constraint, itself
    for j in range(k):
        for i in [0,1]:
            evaluation_site = x[2*(j+1)-i]
            label = y[2*(j+1)-i - 1]    # ~~~ -1 because y_train was not augmented in the way that x_train was
            constraints.append( T[j]-T[j+1] <= theta[j] - nu[j] - (h[j+1]-h[j])*evaluation_site - label*(tau[j+1]-tau[j]) )
            constraints.append( theta[j] - nu[j] - (h[j+1]-h[j])*evaluation_site - label*(tau[j+1]-tau[j]) <= T[j+1]-T[j] )
    #
    # ~~~ Solve it
    problem = cvx.Problem( cvx.Minimize(eps), constraints )
    problem.solve(solver=cvx.SCS)

plt.plot( tau.value, h.value, label="Fitted Model" )
plt.scatter( x_train[1:m+1], y_train, label="Training Data", color="green" )
plt.legend()
plt.xlim([-1,1])
plt.grid()
plt.tight_layout()
plt.show()
