"""
Implement the unusual method training method involving quadratic programming
"""

import torch
from torch import nn
from near_optimal.gradient_univar import spline
import cvxpy as cp
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import root_scalar
from tqdm.auto import tqdm, trange
from matplotlib import pyplot as plt
from quality_of_life.my_plt_utils import points_with_curves
from quality_of_life.my_base_utils import support_for_progress_bars, my_warn
from quality_of_life import my_cvx_utils as mcu

#
# ~~~ Compute the vectors b_j for which we demand the constraint |a_j^Tz| \leq |b_j^Tz|
def build_b_j(x,j):
    #
    # ~~~ Use 1-indexing, assuming that the given j is in zero-indexing to begin with
    assert len(x)%2==0
    k = len(x)//2
    assert j >= 0 and j <= k-1
    j += 1                  # ~~~ so that the formulas from the paper like x_{2j+1} are accurate
    x_2jp2 =  x[2*j+2-1]    # ~~~ x_{2*j+2}
    x_2jp1 =  x[2*j+1-1]    # ~~~ x_{2*j+1}
    x_2j   =  x[2*j-1]      # ~~~ x_{2*j}
    x_2jm1 =  x[2*j-1-1]    # ~~~ x_{2*j-1}
    #
    # ~~~ Compute the non-zero coordinates of the vector b_j
    b_j = torch.zeros_like(x)
    b_j[2*j+2-1] =  (x_2jp1 - x_2j) / (x_2jp2 - x_2jp1) # ~~~ b^{(j)}_{2j+2}
    b_j[2*j+1-1] = -(x_2jp1 - x_2j) / (x_2jp2 - x_2jp1) # ~~~ b^{(j)}_{2j+1}
    b_j[2*j-1]   = -(x_2jp1 - x_2j) / (x_2j - x_2jm1)   # ~~~ b^{(j)}_{2j}
    b_j[2*j-1-1] =  (x_2jp1 - x_2j) / (x_2j - x_2jm1)   # ~~~ b^{(j)}_{2j-1}
    return b_j

#
# ~~~ Compute the vectors a_j for which we demand the constraint |a_j^Tz| \leq |b_j^Tz|
def build_a_j(x,j):
    #
    # ~~~ Use 1-indexing, assuming that the given j is in zero-indexing to begin with
    assert len(x)%2==0
    k = len(x)//2
    assert j >= 0 and j <= k-1
    j += 1                  # ~~~ so that the formulas from the paper like x_{2j+1} are accurate
    x_2jp2 =  x[2*j+2-1]    # ~~~ x_{2*j+2}
    x_2jp1 =  x[2*j+1-1]    # ~~~ x_{2*j+1}
    x_2j   =  x[2*j-1]      # ~~~ x_{2*j}
    x_2jm1 =  x[2*j-1-1]    # ~~~ x_{2*j-1}
    #
    # ~~~ Compute the non-zero coordinates of the vector a_j
    a_j = torch.zeros_like(x)
    a_j[2*j+2-1] =  (x_2jp1 - x_2j) / (x_2jp2 - x_2jp1)     # ~~~ a^{(j)}_{2j+2}
    a_j[2*j+1-1] = -(x_2jp1 - x_2j) / (x_2jp2 - x_2jp1) - 2 # ~~~ a^{(j)}_{2j+1}
    a_j[2*j-1]   =  (x_2jp1 - x_2j) / (x_2j - x_2jm1) + 2   # ~~~ a^{(j)}_{2j}
    a_j[2*j-1-1] = -(x_2jp1 - x_2j) / (x_2j - x_2jm1)       # ~~~ a^{(j)}_{2j-1}
    return a_j

#
# ~~~ Solve t^{a/b} + mse_penalty*t**2 == lb_on_min for t... from minimizing t^a + mse_penalty*MSE subject to |y-z|_{\ell^\infty} \leq t^b
def deduce_lower_bound_on_ERM( lb_on_min, mse_penalty, a, b, upper_bound_on_primal, hard_fail=False ):
    """
    Assume "lb_on_min" is a lower bound on "min_{z,t}( t^a + C*MSE(z,y) S.T. \|z-y\|_{\ell^\infty} \leq t^b and other constraints on z)"
    Notice that we have \|z-y\|_{\ell^\infty}==t^b at the optimum. Then, t^a==(t^b)^{a/b}==\|z-y\|_{\ell^\infty}^{b/a}, and so
    min_{z,t}(the above) == \min_z( \|z-y\|_{\ell^\infty}^{a/b} + C*MSE(z,y) S.T. aforementioned other constraints on z).
    Using the facts that:
     - MSE(z,y) < =\|z-y\|_{\ell^\infty}
     - f(x) := x^{a/b} + C*x is an increasing bijection (0,\infty)\to(0,\infty)
     - f^{-1}(x) is thus also an increasing bijection (0,\infty)\to(0,\infty)
    We obtain lb_on_min <= f(opt) from the first two facts,
    where we introduce opt := \min_z(|z-y\|_{\ell^\infty}^{a/b} S.T. aforementioned other constraints on z)
    and finally lb_on_min <= f(opt) \implies f^{-1}(lb_on_min) <= opt thanks to the last fact.
    Therefore, this function returns f^{-1}(lb_on_min).
    """
    if lb_on_min <= 0: return 0.
    f = lambda t: t**(a/b) + mse_penalty*t**2 - lb_on_min    # ~~~ which we will solve for a lower bound t on |y-z|_{\ell^\infty}
    lower_bound_on_primal = lb_on_min**(b/a) if mse_penalty==0 else root_scalar( f=f, bracket=[0,upper_bound_on_primal] ).root
    msg = f"The supposed lower bound {lower_bound_on_primal} on the primal min is larger than the supplied upper bound {upper_bound_on_primal} (this is mathematically incorrect, implying something is awry)."
    if lower_bound_on_primal > upper_bound_on_primal:
        if hard_fail: raise RuntimeError(msg)
        else: my_warn(msg)
    return lower_bound_on_primal

class DualSpline(spline):
    def __init__( self, x, y, eps=1e-6, lr=1e-2 ):
        super().__init__(x,y)   # ~~~ stores y as self.z
        x = self.x
        self.y = self.z.detach().clone()
        self.lamb = torch.randn(self.k-1).to( device=x.device, dtype=x.dtype )**2
        self.a = torch.stack( [ build_a_j(x,j) for j in range(self.k-1) ] )
        self.b = torch.stack( [ build_b_j(x,j) for j in range(self.k-1) ] )
        self.bbt_minus_aat = torch.stack([ torch.outer(self.b[j],self.b[j]) - torch.outer(self.a[j],self.a[j]) for j in range(self.k-1) ])
        self.t = 0  # ~~~ iterations completed thus far of the Frank-Wolfe algorithm
        self.lr = lr
        self.lower_bounds = x_train[ 2*(torch.arange(k-1)+1)-1 ].squeeze()
        self.upper_bounds = x_train[ 2*(torch.arange(k-1)+1)   ].squeeze()
        self.lower_bound_on_primal = 0.
        self.upper_bound_on_primal = np.inf
        self.project()
        self.update_upper_bound_on_primal()
    #
    # ~~~ Modify z in the way that results from projecting \tau_j onto the interval [x_{2j},x_{2j+1}]
    def project( self, tol=1e-10 ):
        if not torch.allclose( self.z, torch.zeros_like(self.z) ):  # ~~~ prevent a common failure case (but not all possible failure cases)
            with torch.no_grad():
                tau = self.compute_break_points()
                tau.clamp_( min=self.lower_bounds, max=self.upper_bounds )
                self.compute_slopes_and_intercepts()
                a = self.slopes[0]
                b = self.intercepts[0]
                c = self.slopes.diff()
                tau = torch.where( c.abs()>tol, tau, self.c )
                for j in range(self.k):
                    for i in [1,0]:
                        j += 1
                        index_2j_minus_i = 2*j-i - 1
                        evaluation_site = self.x[index_2j_minus_i]  # ~~~ x_{2j-i}
                        j -= 1
                        self.z.data[index_2j_minus_i] = a*evaluation_site + b + sum( c[ell]*(evaluation_site-tau[ell]) for ell in range(j) )
    #
    # ~~~ Project z onto the box constraint set \|z-y\|_{\ell^\infty}\leq\eta
    def ell_infty_projection( self, eta=0.1 ):
        with torch.no_grad():
            self.z.clamp_( min=self.y-eta, max=self.y+eta )
    #
    # ~~~ Save the best upper bound which has been seen seen thus far available
    def update_upper_bound_on_primal( self, tol=1e-10 ):
        with torch.no_grad():
            if self.compute_violation().min().item() >= 1 - tol: # ~~~ if the constraints are satisfied (up to numerical tolerance)
                current_upper_bound = (self.z-self.y).abs().max().item()
                if current_upper_bound < self.upper_bound_on_primal:
                    self.upper_bound_on_primal = current_upper_bound
                    self.best_z = self.z.data.clone()
    #
    # ~~~ Solve the dual problem of "minimize MSE(z,y) subject to (a[i].T@z)**2 - (b[i].T@z)**2 + breakpoint_reg <= 0 for all i = 1,...,k-1"
    def solve_dual_of_mse_minimization( self, *args, breakpoint_reg=0., weighted_mean=False, **kwargs ):    # ~~~ originally had an `mse_penalty` argument but, empirically, that appeared to have no effect
        #
        # ~~~ Setup
        aat_minus_bbt = -self.bbt_minus_aat.cpu().numpy()   # ~~~ shape (self.k-1, 2*self.k, 2*self.k)
        m = len(self.y)
        #
        # ~~~ Whether to use straight MSE or solve for the weights that give the best dual lower bound, as in the computation of what's called b^* in the paper
        if weighted_mean:   # ~~~ then this method computes what is called $b^*$ in the paper
            w = cp.Variable(m)
            constraints = [ w>=0, cp.sum(w)==1 ]
        else:
            constraints = []
        #
        # ~~~ Build and solve problem
        H_o =  (1/m)*np.eye(m)                if not weighted_mean else  cp.diag(w)
        c_o = -(1/m)*self.y.cpu().numpy()     if not weighted_mean else -cp.multiply( w, self.y.cpu().numpy() )
        d_o =  (1/m)*(self.y**2).sum().item() if not weighted_mean else  cp.sum(cp.multiply( w, self.y.cpu().numpy()**2 ))
        problem, _, z = mcu.solve_dual_of_QCQP( H_o, c_o, d_o, H_I=aat_minus_bbt, c_I=(self.k-1)*[np.zeros(m)], d_I=(self.k-1)*[breakpoint_reg], *args, constraints=constraints, **kwargs )
        #
        # ~~~ Process results
        self.z.data = torch.from_numpy(z)
        self.problem = problem  # ~~~ for reference if diagnostics are necessary
        self.project()
        self.update_upper_bound_on_primal()
        dual_max = np.sqrt(problem.objective.value)
        self.lower_bound_on_primal = max( self.lower_bound_on_primal, dual_max )
        return dual_max
    #
    # ~~~ Same as the above, but with more options that I experimented with (including both eqations (9) and (10) from the paper as special cases)
    def solve_dual_of_mse_minimization_with_more_options(
                self,
                *args,
                mse_penalty           = 0,
                t_squared_objective   = False,
                t_squared_constraint  = False,
                breakpoint_reg        = 0,
                non_negative_epigraph = True,
                weighted_mean         = False,
                z_squared_constraint  = True,
                **kwargs
            ):
        return self.the_kitchen_sink(
                *args,
                mse_penalty           = mse_penalty,
                t_squared_objective   = t_squared_objective,
                t_squared_constraint  = t_squared_constraint,
                breakpoint_reg        = breakpoint_reg,
                non_negative_epigraph = non_negative_epigraph,
                weighted_mean         = weighted_mean,
                z_squared_constraint  = z_squared_constraint,
                method                = "solve_dual_of_QCQP",
                **kwargs
            )
    #
    # ~~~ Instead of solving the dual, solve the semi-definite relaxation (this approach is particular to QCQP's, whereas the dual program is much more general)
    def solve_rank_relaxation_of_mse_minimization_with_options(
                self,
                *args,
                mse_penalty           = 0,
                t_squared_objective   = False,
                t_squared_constraint  = False,
                breakpoint_reg        = 0,
                non_negative_epigraph = True,
                weighted_mean         = False,
                z_squared_constraint  = True,
                **kwargs
            ):
        return self.the_kitchen_sink(
                *args,
                mse_penalty           = mse_penalty,
                t_squared_objective   = t_squared_objective,
                t_squared_constraint  = t_squared_constraint,
                breakpoint_reg        = breakpoint_reg,
                non_negative_epigraph = non_negative_epigraph,
                weighted_mean         = weighted_mean,
                z_squared_constraint  = z_squared_constraint,
                method                = "solve_rank_relaxation_of_QCQP",
                **kwargs
            )
    #
    # ~~~ Highly general wrapper that includes *many* different solvers as special cases
    def the_kitchen_sink(
                self,
                *args,
                mse_penalty           = 0,
                t_squared_objective   = False,
                t_squared_constraint  = False,
                breakpoint_reg        = 0,
                non_negative_epigraph = True,
                weighted_mean         = False,
                z_squared_constraint  = True,
                method,
                **kwargs
            ):
        #
        # ~~~ Setup
        if t_squared_constraint and not (t_squared_objective or non_negative_epigraph): my_warn("Minimizing t subject to |y_j-z_j|^2 <= t^2 ain't good...")
        aat_minus_bbt = -self.bbt_minus_aat.cpu().numpy()   # ~~~ shape (self.k-1, 2*self.k, 2*self.k)
        m = len(self.y)
        y_np = self.y.cpu().numpy().flatten()
        if mse_penalty==0:
            if weighted_mean: my_warn("`weighted_mean=True` will be ignored because `mse_penalty==0`.")
            weighted_mean = False
        #
        # ~~~ Whether to use straight MSE or solve for the weights that give the best dual lower bound, as in the computation of what's called b^* in the paper
        if weighted_mean:
            w = cp.Variable(m)
            constraints = [ w >= 0, cp.sum(w) == mse_penalty ]
        else:
            w = (mse_penalty/m) * np.ones(m)
            constraints = []
        #
        # ~~~ Build objective function
        concatenate = cp.hstack   if weighted_mean else np.concatenate
        diag        = cp.diag     if weighted_mean else np.diag
        multiply    = cp.multiply if weighted_mean else np.multiply
        H_o = diag(concatenate([ w, [1. if t_squared_objective else 0.] ]))
        c_o = concatenate([ -multiply(w,y_np), [0. if t_squared_objective else 1/2] ])
        d_o = sum(multiply( w, y_np**2 ))
        #
        # ~~~ Build constraints
        if z_squared_constraint:
            #
            # ~~~ Constraints (a[i].T@z)**2 - (b[i].T@z)**2 + breakpoint_reg <= 0 and (z_j-y_j)**2 - t^b \leq 0
            H_I = np.concatenate([
                    np.pad( aat_minus_bbt, ( (0,0), (0,1), (0,1) ) ),   # ~~~ pad the "actual constraints" with zero for the epigraph variable
                    np.stack([
                            np.diag( j*[0.] + [1.] + (m-j-1)*[0.] + [-1. if t_squared_constraint else 0.]) 
                            for j in range(m)
                        ])
                ])
            c_I = np.vstack([
                    np.zeros(( self.k-1, m+1 )),
                    np.hstack(( np.diag(-y_np), -(not t_squared_constraint)*np.ones((m,1))/2 ))
                ])
            d_I = np.concatenate([
                    breakpoint_reg*np.ones(self.k-1),
                    y_np**2
                ])
            b = (t_squared_constraint + 1) / 2
        else:
            #
            # ~~~ Simon suggested this instead of the non-convex quadratic epigraph constraint that I was using, as in eq'n (9) in the paper (unless revisions have resulted in this number changing)
            H_I = np.concatenate([
                    np.pad( aat_minus_bbt, ( (0,0), (0,1), (0,1) ) ),   # ~~~ pad the "actual constraints" with zero for the epigraph variable
                    np.zeros(( 2*m, m+1, m+1 ))
                ])
            if t_squared_constraint:
                for j in range(2*m):
                    H_I[ self.k+j-1, -1, -1 ] = -1.
            c_I = np.vstack([
                    np.zeros(( self.k-1, m+1 )),
                    np.hstack([ np.eye(m), -(not t_squared_constraint)*np.ones((m,1)) ])/2,
                    np.hstack([-np.eye(m), -(not t_squared_constraint)*np.ones((m,1)) ])/2
                ])
            d_I = np.concatenate([ breakpoint_reg*np.ones(self.k-1), -y_np, y_np ])
            b = t_squared_constraint + 1
        #
        # ~~~ Optionally, add t >= 0 constraint if desired (I don't think it makes any difference?)
        if non_negative_epigraph:
            H_I = np.concatenate([ H_I, np.zeros(( 1, m+1, m+1 )) ])
            c_I = np.concatenate([ c_I, np.array(m*[0.] + [-1.]).reshape(1,m+1) ])
            d_I = np.concatenate([ d_I, [0.] ])
        #
        # ~~~ Solve the problem and process the results
        self.problem, _, zt = getattr(mcu,method)( H_o, c_o, d_o, H_I=H_I, c_I=c_I, d_I=d_I, *args, constraints=constraints, **kwargs )
        self.z.data = torch.from_numpy(zt[:-1])
        self.project()
        self.update_upper_bound_on_primal()
        lower_bound = deduce_lower_bound_on_ERM(
                lb_on_min = self.problem.objective.value,
                mse_penalty = mse_penalty,
                a = t_squared_objective + 1,
                b = b,
                upper_bound_on_primal = self.upper_bound_on_primal
            )
        self.lower_bound_on_primal = max( self.lower_bound_on_primal, lower_bound )
        return lower_bound
    #
    # ~~~ Cursory implementation of yet another option: minimizing some meaningless function subject to \|z-y\|_2^2<=noise and (a[i].T@z)**2 - (b[i].T@z)**2 <= 0
    def solve_dual_of_quadratic_feasibility_program( self, noise=0.1, *args, **kwargs  ):
        #
        # ~~~ Set the objective function to be identically equal to 1.
        m = len(self.y)
        H_o = np.eye(m)
        c_o = np.zeros(m)
        d_o = 1.
        #
        # ~~~ Set the inequality constraints
        aat_minus_bbt = -self.bbt_minus_aat.cpu().numpy()                   # ~~~ shape (self.k-1, m, m)
        H_I = np.concatenate([ aat_minus_bbt, np.eye(m)[np.newaxis,:]/m ])  # ~~~ shape ( self.k,  m, m)
        c_I = (self.k-1)*[np.zeros(m)] + [-self.y.cpu().numpy()/m]
        d_I = (self.k-1)*[0.] + [ (self.y.cpu()**2).mean().numpy() - noise ]
        #
        # ~~~ Solve the dual
        self.problem, _, z = mcu.solve_dual_of_QCQP( H_o, c_o, d_o, H_I=H_I, c_I=c_I, d_I=d_I, *args, **kwargs )
        self.z.data = torch.from_numpy(z)
    #
    # ~~~ Minimize t+mse_penalty*MSE(y,z) subject to |s_jx + c_j - y| \leq t for both data pairs (x,y), for j=1,...,k, and subject to p_j(s_1,...,s_k,c_1,...,c_k) \leq 0
    def D_kappa( self, *args, M=0, kappa=0, mse_penalty=0, t_squared_objective=True, announce_eigenvalues=True, **kwargs ):
        #
        # ~~~ Define objects of the correct size
        k = self.k
        m = len(self.y)
        H_o = np.diag(np.concatenate([ np.zeros(m), [1. if t_squared_objective else 0.] ])) #np.zeros((m+1,m+1))
        c_o = np.concatenate([ np.zeros(m), [0. if t_squared_objective else 1/2] ]) #np.array( 2*k*[0.] + [1/2] )
        d_o = 0
        H_I = np.zeros(( k-1+2*m, 2*k+1, 2*k+1 ))
        c_I = np.zeros(( k-1+2*m, 2*k+1 ))
        d_I = np.zeros( k-1+2*m )
        #
        # ~~~ Penalize MSE
        for j in range(k):
            for i in [1,0]:
                j += 1  # ~~~ use 1-based indexing j=1,...,k
                index_2j_minus_i = (2*j-i)-1
                j -= 1  # ~~~ return to 0-based indexing
                #
                # ~~~ Add (mse_penalty/m)*( s_j*x_{2j-i} + c_j - y_{2j-i} )^2 to the objective function
                evaluation_site  = self.x[index_2j_minus_i].item() + M  # ~~~ == x_{2j-i} + M
                training_label   = self.y[index_2j_minus_i].item()      # ~~~ == y_{2j-i}
                H_o[j,j]        += (mse_penalty/m) * evaluation_site**2 # ~~~ (mse_penalty/m) * x_{2j-i}^2 * s_j^2
                H_o[k+j,j]      += (mse_penalty/m) * evaluation_site
                H_o[j,k+j]      += (mse_penalty/m) * evaluation_site
                H_o[k+j,k+j]    += mse_penalty/m                        # ~~~ (mse_penalty/m) * c_j^2
                c_o[j]          -= (mse_penalty/m) * training_label * evaluation_site   # ~~~ -2(mse_penalty/m) * x_{2j-i} * y_{2j-i} * s_j
                c_o[k+j]        -= (mse_penalty/m) * training_label     # ~~~ -2(mse_penalty/m) * y_{2j-i} * c_j
                d_o             += (mse_penalty/m) * training_label**2  # ~~~ y_{2j-i}^2
        #
        # ~~~ Safety feature
        for j in range(k-1):
            j += 1  # ~~~ use 1-based indexing j=1,...,k-1
            delta_over_2 = (self.x[(2*j)-1] - self.x[(2*j-1)-1])/2          # ~~~ == (x_{2j} - x_{2j-1})/2
            shifted_midpoint = (self.x[(2*j+1)-1] + self.x[(2*j)-1])/2 + M  # ~~~ == (x_{2j+1} + x_{2j})/2 + M
            if abs(shifted_midpoint) < 1e-8:
                my_warn("A midpoint of the shifted data is approximately zero. Consider toggling the value of M.")
                break
        #
        # ~~~ Build the "actual constraints," i.e., the constraints on the locations of breakpoints
        for j in range(k-1):
            j += 1  # ~~~ use 1-based indexing j=1,...,k-1
            delta_over_2 = (self.x[(2*j)-1] - self.x[(2*j-1)-1]).item()/2           # ~~~ == (x_{2j} - x_{2j-1})/2
            shifted_midpoint = 1 #(self.x[(2*j+1)-1] + self.x[(2*j)-1]).item()/2 + M   # ~~~ == (x_{2j+1} + x_{2j})/2 + M
            j -= 1  # ~~~ return to 0-based indexing
            A = 1 - (delta_over_2/shifted_midpoint)**2
            H_I[ j, j, j ]     =  A
            H_I[ j, j, j+1 ]   = -A
            H_I[ j, j+1, j ]   = -A
            H_I[ j, j+1, j+1 ] =  A
            B = 1/shifted_midpoint
            H_I[ j, j, k+j ]     =  B
            H_I[ j, j, k+j+1 ]   = -B
            H_I[ j, j+1, k+j ]   = -B
            H_I[ j, j+1, k+j+1 ] =  B
            H_I[ j, k+j, j ]     =  B
            H_I[ j, k+j+1, j ]   = -B
            H_I[ j, k+j, j+1 ]   = -B
            H_I[ j, k+j+1, j+1 ] =  B
            C = 1/shifted_midpoint**2
            H_I[ j, k+j, k+j ]     =  C
            H_I[ j, k+j, k+j+1 ]   = -C
            H_I[ j, k+j+1, k+j ]   = -C
            H_I[ j, k+j+1, k+j+1 ] =  C
            smallest_eigenvalue, corresponding_eigenvector = eigsh( H_I[j,:,:], k=1, which='SA' )
            H_I[ j, :, : ] += kappa*smallest_eigenvalue*np.eye(2*k+1)
            if announce_eigenvalues: print(f"The Hessian of quadratic constraint {j+1}/{k-1} has smallest eigenvalue {smallest_eigenvalue[0]}.")
        #
        # ~~~ Build the epigraph constraints
        for j in range(k):
            for i in [1,0]:
                j += 1  # ~~~ use 1-based indexing j=1,...,k
                index_2j_minus_i = (2*j-i)-1
                j -= 1  # ~~~ return to 0-based indexing
                #
                # ~~~ Add the constraint s_j*(x+M) + c_j - y - t \leq 0
                evaluation_site = self.x[index_2j_minus_i].item()  # ~~~ == x_{2j-i}
                training_label  = self.y[index_2j_minus_i].item()  # ~~~ == y_{2j-i}
                c_I[ k-1+index_2j_minus_i, j   ] = (evaluation_site+M)/2
                c_I[ k-1+index_2j_minus_i, k+j ] = 1/2
                c_I[ k-1+index_2j_minus_i, -1  ] = -1/2
                d_I[ k-1+index_2j_minus_i ]      = -training_label
                #
                # ~~~ Add the constraint -s_j*(x+M) - c_j + y - t \leq 0
                c_I[ k-1+m+index_2j_minus_i, j   ] = -(evaluation_site+M)/2
                c_I[ k-1+m+index_2j_minus_i, k+j ] = -1/2
                c_I[ k-1+m+index_2j_minus_i, -1  ] = -1/2
                d_I[ k-1+m+index_2j_minus_i ]      = training_label
        #
        # ~~~ Solve the dual
        problem, _, s_c_t = mcu.solve_dual_of_QCQP( H_o, c_o, d_o, H_I=H_I, c_I=c_I, d_I=d_I, *args, **kwargs )
        s, c, t = np.array_split( s_c_t, [k,2*k] )
        for j in range(k):
            for i in [1,0]:
                j += 1  # ~~~ use 1-based indexing j=1,...,k
                appropriate_index = (2*j-i)-1
                j -= 1  # ~~~ return to 0-based indexing
                with torch.no_grad(): self.z.data[appropriate_index] = s[j]*self.x[appropriate_index] + c[j]
        return np.sqrt(abs(problem.objective.value)) if t_squared_objective else problem.objective.value
    #
    # ~~~ Minimize t+mse_penalty*MSE(y,z) subject to |s_jx + c_j - y| \leq t for both data pairs (x,y), for j=1,...,k, and subject to p_j(s_1,...,c_k) + \kappa\|s_1,...,s_k\|^2 \leq 0 which is equivalent to (s_1,...,c_k) lying in the eigenspace of p_j's Hessian's smallest eigenvalue, which we enforce as a linear equality constraint
    def P_kappa_1( self, *args, M=0, mse_penalty=0, t_squared_objective=False, announce_eigenvalues=True, **kwargs ):
        #
        # ~~~ Define objects of the correct size
        k = self.k
        s = cp.Variable(k)
        c = cp.Variable(k)
        t = cp.Variable()
        alpha = cp.Variable(k-1)
        epigraph_objective = t**2 if t_squared_objective else t
        constraints = []
        #
        # ~~~ Safety feature
        for j in range(k-1):
            j += 1  # ~~~ use 1-based indexing j=1,...,k-1
            delta_over_2 = (self.x[(2*j)-1] - self.x[(2*j-1)-1]).item()/2           # ~~~ == (x_{2j} - x_{2j-1})/2
            shifted_midpoint = (self.x[(2*j+1)-1] + self.x[(2*j)-1]).item()/2 + M   # ~~~ == (x_{2j+1} + x_{2j})/2 + M
            if abs(shifted_midpoint) < 1e-8:
                my_warn("A midpoint of the shifted data is approximately zero. Consider toggling the value of M.")
                break
        #
        # ~~~ Build the "actual constraints"
        for j in range(k-1):
            j += 1  # ~~~ use 1-based indexing j=1,...,k-1
            delta_over_2 = (self.x[(2*j)-1] - self.x[(2*j-1)-1]).item()/2           # ~~~ == (x_{2j} - x_{2j-1})/2
            shifted_midpoint = (self.x[(2*j+1)-1] + self.x[(2*j)-1]).item()/2 + M   # ~~~ == (x_{2j+1} + x_{2j})/2 + M
            j -= 1  # ~~~ return to 0-based indexing
            A = 1 - (delta_over_2/shifted_midpoint)**2
            B = 1/shifted_midpoint
            C = 1/shifted_midpoint**2
            H = np.array([
                    [  A, -A,  B, -B ],
                    [ -A,  A, -B,  B ],
                    [  B, -B,  C, -C ],
                    [ -B,  B, -C,  C ]
                ])
            smallest_eigenvalue, corresponding_eigenvector = eigsh( H, k=1, which='SA' )
            corresponding_eigenvector = corresponding_eigenvector.flatten()
            if announce_eigenvalues: print(f"The Hessian of quadratic constraint {j+1}/{k-1} has smallest eigenvalue {smallest_eigenvalue[0]}.")
            constraints += [
                s[j]   == alpha[j]*corresponding_eigenvector[0],
                s[j+1] == alpha[j]*corresponding_eigenvector[1],
                c[j]   == alpha[j]*corresponding_eigenvector[2],
                c[j+1] == alpha[j]*corresponding_eigenvector[3]
            ]
        #
        # ~~~ Build the epigraph constraints
        for j in range(k):
            for i in [1,0]:
                j += 1  # ~~~ use 1-based indexing j=1,...,k
                index_2j_minus_i = (2*j-i)-1
                j -= 1  # ~~~ return to 0-based indexing
                evaluation_site = self.x[index_2j_minus_i].item()  # ~~~ == x_{2j-i}
                training_label  = self.y[index_2j_minus_i].item()  # ~~~ == y_{2j-i}
                error = s[j]*(evaluation_site+M) + c[j] - training_label
                #
                # ~~~ Add the constraint s_j*(x+M) + c_j - y - t \leq 0
                constraints.append( error - t <= 0 )
                #
                # ~~~ Add the constraint -s_j*(x+M) - c_j + y - t \leq 0                
                constraints.append( -error - t <= 0 )
                #
                # ~~~ If penalizing the MSE, add this error squared to the objective
                if mse_penalty>0: epigraph_objective += (mse_penalty/m)*error**2
        #
        # ~~~ Solve it
        objective = cp.Minimize( epigraph_objective )
        problem = cp.Problem( objective, constraints )
        problem.solve( *args, **kwargs )
        return problem
    #
    # ~~~ 
    def PGD_step(self):
        #
        # ~~~ Compute the gradient of F(\lambda)
        self.Q = torch.ones_like(self.y).diag() - (self.lamb.reshape(-1,1,1)*self.bbt_minus_aat).sum(dim=0) # ~~~ Q(\lambda) = I - \sum_{j=1}^{k-1} \lambda_j (b_j b_j^T - a_j a_j^T)
        z = torch.linalg.solve( self.Q, self.y )            # ~~~ z = Q(\lambda)^{-1}y 
        g = ((self.a@z)**2 - (self.b@z)**2).cpu().numpy()   # ~~~ \grad_\lambda F(\lambda)
        objective_before_update = -torch.inner( self.z, self.y ) + (self.y**2).sum()
        self.z.data = z
        self.lamb += self.lr*g
        # print( self.lamb, end="\n" )
        #
        # ~~~ Project onto the constraint set
        s = cp.Variable(self.k-1)
        objective = cp.Minimize( cp.sum_squares( s - self.lamb.double().cpu().numpy() ))
        bbt_minus_aat = self.bbt_minus_aat.cpu().numpy()
        R = sum(s[i] * bbt_minus_aat[i] for i in range(self.k-1))
        constraints = [
                s >= 0,
                R << (1-self.eps)*np.eye(2*self.k)
            ]
        problem = cp.Problem( objective, constraints )
        problem.solve()
        self.lamb = torch.from_numpy(s.value).to( device=self.lamb.device, dtype=self.lamb.dtype )
        return objective_before_update.detach().cpu().item()
    #
    # ~~~ 
    def frank_wolfe_step(self):
        #
        # ~~~ Compute the gradient of F(\lambda)
        self.Q = torch.ones_like(self.y).diag() - (self.lamb.reshape(-1,1,1)*self.bbt_minus_aat).sum(dim=0) # ~~~ Q(\lambda) = I - \sum_{j=1}^{k-1} \lambda_j (b_j b_j^T - a_j a_j^T)
        z = torch.linalg.solve( self.Q, self.y )            # ~~~ z = Q(\lambda)^{-1}y 
        g = ((self.a@z)**2 - (self.b@z)**2).cpu().numpy()   # ~~~ \grad_\lambda F(\lambda)
        objective_before_update = -torch.inner( self.z, self.y ) + (self.y**2).sum()
        g *= self.lr
        self.z.data = z
        #
        # ~~~ Solve the Frank-Wolfe subproblem to find a better update direction than the gradient
        s = cp.Variable(self.k-1)
        objective = cp.Maximize( g@s )
        bbt_minus_aat = self.bbt_minus_aat.cpu().numpy()
        R = sum(s[i] * bbt_minus_aat[i] for i in range(self.k-1))
        constraints = [
                s >= self.eps,
                R << (1-self.eps)*np.eye(2*self.k)
            ]
        problem = cp.Problem(objective, constraints)
        duality_gap = problem.solve()/self.lr
        if duality_gap == float("inf"):
            # print("decreasing learning rate")
            self.lr /= (1+0.1)
        else:
            # print("increasing learning rate")
            self.lr *= (1+0.001)
            alpha = 2/(self.t+2)
            self.lamb = (1-alpha)*self.lamb + alpha*torch.from_numpy(s.value).to( device=self.lamb.device, dtype=self.lamb.dtype )
            self.t += 1
        return objective_before_update, duality_gap

#
# ~~~ Config
torch.manual_seed(2025)
torch.set_default_dtype(torch.double)
k = 15
m =  2*k
f = lambda x: torch.sin(2*torch.pi*x)
noise_level = 0.1
x_train = torch.linspace(-1,1,m)
y_train = f(x_train) + noise_level*torch.randn_like(x_train)
x_test = torch.linspace(-1,1,1001)
y_test = f(x_test)

if __name__ == "__main__":
    v = DualSpline( x_train, y_train )
    N = None
    noise = None
    # #
    # # ~~~ Try gradient descent on the problem \min_z \max_\ell -\langle z,a_\ell \rangle*\langle z,b_\ell \rangle subject to \|z-y\|_\infty \leq \eta
    # v.ell_infty_projection(eta=noise)
    # optimizer = torch.optim.Adam( v.parameters(), lr=1e-3 )
    # gif = GifMaker()
    # fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), title="Minimize the Violation Subject to an \ell^\infty Constraint", show=False )
    # gif.capture()
    # with support_for_progress_bars():
    #     for _ in tqdm(range(2000)):
    #         predictions = v.z   # == v(x_train)
    #         loss = ( (v.a@predictions)**2 - (v.b@predictions)**2 ).max()
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         v.ell_infty_projection(eta=noise)
    #         if (_+1)%10==0:
    #             fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), title="Minimize the Violation Subject to an \ell^\infty Constraint", show=False, fig=fig, ax=ax )
    #             gif.capture()
    #     fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), title="Minimize the Violation Subject to an \ell^\infty Constraint", fig=fig, ax=ax, show=False )
    #     gif.develop()
    #
    # ~~~ Solve using the S-lemma
    b_star = v.solve_dual_of_mse_minimization(weighted_mean=True)
    if N is None:
        val = v.solve_dual_of_mse_minimization( solver="SCS", eps=1e-6 )
        best_z = v.z.data.clone()
    if N is not None:
        best_error = float("inf")
        with support_for_progress_bars():
            pbar = tqdm( desc="Solving the Dual Problem Using Frank-Wolfe", total=N, ascii=' >=' )
            for _ in range(N):
                F, duality_gap = v.frank_wolfe_step()
                _ = pbar.update()
                with torch.no_grad():
                    errors = v(x_train) - y_train
                    mse = (errors**2).mean()
                    max_error = errors.abs().max()
                    pbar.set_postfix({
                            "mse" : f"{mse:<4.4f}",
                            "max_error" : f"{max_error:<4.4f}",
                            "F(lambda)" : f"{F:<4.4f}",
                            "gap" : f"{duality_gap:<4.4f}"
                        })
                    if max_error < best_error:
                        best_error = max_error
                        best_z = v.z.data.clone()
        pbar.close()
    v.z.data = best_z
    with torch.no_grad(): pred = v(x_train)
    max_abs_error = (pred-y_train).abs().max().item()
    mean_sq_error = ((pred-y_train)**2).mean().item()
    suboptimality_ratio = max_abs_error/b_star
    fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), show=False, title=f"The Result of Some Quadratic Program (sub-optimality ratio: {suboptimality_ratio:.4f})" )
    with torch.no_grad():
        nodes = v.compute_break_points()
        ax.scatter( nodes, v(nodes), color="blue", alpha=0.4 )
    plt.show()
    # #
    # # ~~~ Try taking that as the initialization for a ReLU network
    # from near_optimal.PGD_univar import RigorousNet
    # v = RigorousNet(x_train)
    # with torch.no_grad():
    #     slopes = (best_z[1::2] - best_z[::2]) / (x_train[1::2] - x_train[::2])
    #     v.relu_net[0].bias.data = -nodes.reshape(v.relu_net[0].bias.data.shape)
    #     v.a.data = slopes[0]
    #     v.relu_net[-1].weight.data = slopes.diff().reshape(v.relu_net[-1].weight.data.shape)
    #     v.relu_net[-1].bias.fill_( best_z[0] - slopes[0]*x_train[0] )
    # fig, ax = points_with_curves( x=x_train, y=y_train, grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(v,f), show=False, title="MSE Minimization Subject to Constraints on the Location of Breakpoints" )
    # with torch.no_grad():
    #     ax.scatter( nodes, v(nodes.reshape(-1,1)), color="blue", alpha=0.4 )
    # plt.show()
    # #
    # # ~~~ Train it
    # optimizer = torch.optim.Adam( v.parameters(), lr=1e-2 )
    # x_train_vertical = x_train.reshape(-1,1)
    # gif = GifMaker()
    # fig, ax = points_with_curves( x=x_train, y=y_train, grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(v,f), title="MSE Minimization Subject to Constraints on the Location of Breakpoints", show=False )  
    # gif.capture()
    # with support_for_progress_bars():
    #     for _ in trange(10000):
    #         predictinons = v(x_train_vertical)
    #         max_error = (y_train-predictinons).abs().max()
    #         max_error.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         v.project()
    #         if (_+1)%100:
    #             fig, ax = points_with_curves( x=x_train, y=y_train, grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(v,f), title="MSE Minimization Subject to Constraints on the Location of Breakpoints", show=False, fig=fig, ax=ax )
    #             gif.capture()
    #     points_with_curves( x=x_train, y=y_train, grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(v,f), title="MSE Minimization Subject to Constraints on the Location of Breakpoints", fig=fig, ax=ax )
    #     gif.develop()
    #
    # ~~~ Train a neural net for comparison
    model = nn.Sequential(
            nn.Unflatten( dim=-1, unflattened_size=(-1,1) ),
            nn.Linear(1,k-1),
            nn.ReLU(),
            nn.Linear(k-1,1)
        )
    big_model = nn.Sequential(
            nn.Unflatten( dim=-1, unflattened_size=(-1,1) ),
            nn.Linear(1,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,1),
        )
    occasional_model = nn.Sequential(
            nn.Unflatten( dim=-1, unflattened_size=(-1,1) ),
            nn.Linear(1,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,1),
        )
    # gif = GifMaker()
    optimizer = torch.optim.Adam( model.parameters(), lr=1e-2 )
    big_optimizer = torch.optim.Adam( big_model.parameters(), lr=1e-2 )
    occasional_optimizer = torch.optim.Adam( occasional_model.parameters(), lr=1e-2 )
    with support_for_progress_bars():
        for epoch in trange(10000):
            loss = (( model(x_train).flatten() - y_train )**2).mean()
            loss.backward()
            big_loss = (( big_model(x_train).flatten() - y_train )**2).mean()
            big_loss.backward()
            if epoch % 10 == 0:
                occasional_loss = (( occasional_model(x_train).flatten() - y_train )**2).mean()
                occasional_loss.backward()
            for opt in ( optimizer, big_optimizer, occasional_optimizer ):
                opt.step()
                opt.zero_grad()
    fig, ax = points_with_curves(
            x = x_train,
            y = y_train,
            marker_size  = 6,  # ~~~ size of the scatter plot
            curves       = ( big_model, occasional_model, v,        f       ),
            curve_colors = ( "black",   "grey",           "blue",   "green" ),
            curve_marks  = [ "-",       "-",              "-",      "--"    ],
            curve_labels = ( "Large Network Trained with ADAM", "Large Network Trained with ADAM and Early Stopping", "Small Network Trained with Our Method", "Ideal Fit" ),
            ylim = [-1.1,1.1],
            figsize = (12,6),
            show = False,
            title = r"Comparison Between Our Model ($\widehat{C}\approx$" + f"{suboptimality_ratio:.1f}, in blue, {2+2*(k-1)} parameters) versus Larger Networks Trained using ADAM ({sum( p.numel() for p in big_model.parameters() )} parameters)",
            model_fit = False  # ~~~ deactivate default settings
        )
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))  # Get unique labels
    by_label = {}   # Create a dictionary to store handles and line styles for each unique label
    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]  # Find indices for each label
        handle = handles[indices[0]]  # Get the handle for the first occurrence of the label
        line_style = handle.get_linestyle()  # Get the line style
        by_label[label] = (handle, line_style)  # Store handle and line style
    legend_handles = [by_label[label][0] for label in by_label]
    legend_labels = [f"{label}" for label in by_label]  # Include line style in label
    plt.legend( legend_handles, legend_labels, fontsize=8.2, loc="upper right" )
    plt.savefig( "jmlr_fig", dpi=400 )
    plt.show()
    #
    # ~~~ Finer option
    v.solve_dual_of_mse_minimization_with_more_options(mse_penalty=1)
    with torch.no_grad(): pred = v(x_train)
    new_max_abs_error = (pred-y_train).abs().max().item()
    new_mean_sq_error = ((pred-y_train)**2).mean().item()
    new_suboptimality_ratio = new_max_abs_error/b_star
    print("")
    print('The "Slightly more Refined Quadratic Program" may be considered "better" depending on the following stats....')
    print(f"   mean sq error {'down' if new_mean_sq_error<mean_sq_error else 'up'} from {mean_sq_error} to {new_mean_sq_error}")
    print(f"   max abs error {'down' if new_max_abs_error<max_abs_error else 'up'} from {max_abs_error} to {new_max_abs_error} (this is the one our theory cares about)")
    print(f"   sub-optimality ratio {'down' if new_suboptimality_ratio<suboptimality_ratio else 'up'} from {suboptimality_ratio} to {new_suboptimality_ratio} (anything <2 is good)")
    print("")
    fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), show=False, title="Slightly more Refined Quadratic Program (see stdout for stats)" )
    with torch.no_grad():
        nodes = v.compute_break_points()
        ax.scatter( nodes, v(nodes), color="blue", alpha=0.4 )
    plt.show()
