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
    # ~~~ Basic fit
    def fit(self):
        b_star = self.solve_dual_of_mse_minimization(weighted_mean=True)
        self.solve_dual_of_mse_minimization()
        with torch.no_grad(): pred = self(x_train)
        max_abs_error = (pred-y_train).abs().max().item()
        print("")
        print(f"    Sub-optimality ratio: {max_abs_error/b_star}")
        print("")
    #
    # ~~~ Solve the dual problem of "minimize MSE(z,y) subject to (a[i].T@z)**2 - (b[i].T@z)**2 + breakpoint_reg <= 0 for all i = 1,...,k-1"
    def solve_dual_of_mse_minimization( self, breakpoint_reg=0., weighted_mean=False, tol=1e-7  ):    # ~~~ originally had an `mse_penalty` argument but, empirically, that appeared to have no effect
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
        problem, _, z = mcu.solve_dual_of_QCQP( H_o, c_o, d_o, H_I=aat_minus_bbt, c_I=(self.k-1)*[np.zeros(m)], d_I=(self.k-1)*[breakpoint_reg], constraints=constraints, solver="SCS", eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000  )
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
f = lambda x: torch.sin(2*torch.pi*x)*(1-torch.exp(-x**2)) # ~~~ a bit contrived, yes, but only in the interest of giving a rich example
noise_level = 0.1
x_train = torch.linspace(-1,1,m)
x_train = x_train.sign() * x_train.abs().sqrt() # ~~~ make the problem harder by inducing a gap in the middle of the data
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
        val = v.solve_dual_of_mse_minimization()
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
            if epoch % 100 == 0: # ~~~ I'm willing to call this "early stopping" because I did a coarse manual grid search ( % 10, % 20, etc. ) which is equivalent to testing multiple checkpoints and choosing the best one
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
            curve_marks  = [ "--",      (0,(5,5)),         "-",      ":"    ],
            curve_labels = ( "Large Network Trained with ADAM", "Large Network Trained with ADAM and Early Stopping", "Small Network Trained with Our Method", r"$f$" ),
            curve_thicknesses = ( 1.25, 1.25, 1.25, 1.25 ),
            ylim = [-.75,.75],
            figsize = (12,6),
            show = False,
            title = r"Comparison Between Our Model ($\widehat{C}\approx$" + f"{suboptimality_ratio:.1f}, in blue, {2+2*(k-1)} parameters) versus Larger Networks Trained using ADAM ({sum( p.numel() for p in big_model.parameters() )} parameters)",
            model_fit = False  # ~~~ deactivate default settings
        )
    ax.legend( fontsize=14, title_fontsize=16, markerscale=1.5, loc="upper right" )
    plt.savefig( "jmlr_fig", dpi=400 )
    plt.show()
    #
    # ~~~ Finer option
    v.solve_dual_of_mse_minimization_with_more_options( mse_penalty=1, eps_abs=1e-8, eps_rel=1e-8, eps_infeas=1e-11 )
    with torch.no_grad(): pred = v(x_train)
    new_max_abs_error = (pred-y_train).abs().max().item()
    new_mean_sq_error = ((pred-y_train)**2).mean().item()
    new_suboptimality_ratio = new_max_abs_error/b_star
    print("")
    print('The "Slightly more Refined Quadratic Program" may or may not be "better" depending on the following stats....')
    print(f"   mean sq error {'down' if new_mean_sq_error<mean_sq_error else 'up'} from {mean_sq_error} to {new_mean_sq_error}")
    print(f"   max abs error {'down' if new_max_abs_error<max_abs_error else 'up'} from {max_abs_error} to {new_max_abs_error} (this is the one our theory cares about)")
    print(f"   sub-optimality ratio {'down' if new_suboptimality_ratio<suboptimality_ratio else 'up'} from {suboptimality_ratio} to {new_suboptimality_ratio} (anything <2 is good)")
    print("")
    fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), show=False, title="Slightly more Refined Quadratic Program (see stdout for stats)" )
    with torch.no_grad():
        nodes = v.compute_break_points()
        ax.scatter( nodes, v(nodes), color="blue", alpha=0.4 )
    plt.show()
