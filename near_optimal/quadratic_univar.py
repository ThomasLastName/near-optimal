
import torch
from torch import nn
from near_optimal.gradient_univar import spline
from near_optimal.PGD_univar import RigorousNet
import cvxpy as cp
import numpy as np
from scipy.sparse.linalg import eigsh
from tqdm.auto import tqdm, trange
from matplotlib import pyplot as plt
from quality_of_life.my_plt_utils import points_with_curves, GifMaker
from quality_of_life.my_base_utils import support_for_progress_bars, my_warn
from quality_of_life.my_cvx_utils import solve_dual_of_QCQP, solve_rank_relaxation_of_QCQP



### ~~~
## ~~~ Compute the vectors a_j and b_j for which we demand the constraint |a_j^Tz| \leq |b_j^Tz|
### ~~~

def build_b_j(x,j):
    #
    # ~~~ Use 1-indexing, assuming that the given j is in zero-indexing to begin with
    assert len(x)%2==0
    k = len(x)//2
    assert j >= 0 and j <= k-1
    j += 1
    x_2jp2 = x[2*j+2-1]
    x_2jp1 = x[2*j+1-1]
    x_2j   = x[2*j-1]
    x_2jm1 = x[2*j-1-1]
    D_j   =  (x_2jp1 - x_2j)    # ~~~ D_j = x_{2j+1} - x_{2j}, the length of the interval on which a break point is allowed
    d_j   =  (x_2j   - x_2jm1)  # ~~~ d_j = x_{2j} - x_{2j-1}, the length of one of the intervals in which no break point is allowed
    d_jp1 =  (x_2jp2 - x_2jp1)  # ~~~ d_{j+1} = x_{2j+2} - x_{2j+1}, the length of one of the intervals in which no break point is allowed
    #
    # ~~~ Compute the non-zero coordinates of the vector b_j
    b_2jp2 = (D_j/2) / d_jp1
    b_2jp1 = (D_j/2) * (-1) / (d_jp1)
    b_2j   = (D_j/2) * (-1) / (d_j)
    b_2jm1 = (D_j/2) / d_j
    #
    # ~~~ Assign the computed coefficients to the non-zero positions in the vector b_j
    b_j = torch.zeros_like(x)
    b_j[2*j+2-1] = b_2jp2   # ~~~ b^{(j)}_{2j+2}
    b_j[2*j+1-1] = b_2jp1   # ~~~ b^{(j)}_{2j+1}
    b_j[2*j-1]   = b_2j     # ~~~ b^{(j)}_{2j}
    b_j[2*j-1-1] = b_2jm1   # ~~~ b^{(j)}_{2j-1}
    return b_j

def build_a_j(x,j):
    #
    # ~~~ Use 1-indexing, assuming that the given j is in zero-indexing to begin with
    assert len(x)%2==0
    k = len(x)//2
    assert j >= 0 and j <= k-1
    j += 1
    x_2jp2 =  x[2*j+2-1]    # ~~~ x_{2*j+2}
    x_2jp1 =  x[2*j+1-1]    # ~~~ x_{2*j+1}
    x_2j   =  x[2*j-1]      # ~~~ x_{2*j}
    x_2jm1 =  x[2*j-1-1]    # ~~~ x_{2*j-1}
    c_j    =  (x_2j   + x_2jp1)/2   # ~~~ c_j = (x_{2j} + x_{2j+1})/2, the midpoint of the interval where a break point is allowed
    d_j    =  (x_2j   - x_2jm1)     # ~~~ d_j = x_{2j} - x_{2j-1}, the length of one of the intervals in which no break point is allowed
    d_jp1  =  (x_2jp2 - x_2jp1)     # ~~~ d_{j+1} = x_{2j+2} - x_{2j+1}, the length of one of the intervals in which no break point is allowed
    m_j    =  (x_2j   + x_2jm1)/2   # ~~~ m_j = (x_{2j} + x_{2j-1})/2, the midpoint of one of the intervals in which no break point is allowed
    m_jp1  =  (x_2jp2 + x_2jp1)/2   # ~~~ m_{j+1} = (x_{2j+2} + x_{2j+1})/2, the midpoint of one of the intervals in which no break point is allowed
    #
    # ~~~ Compute the non-zero coordinates of the vector a_j
    a_2jp2 =  m_jp1/d_jp1 - 1/2 - c_j/d_jp1
    a_2jp1 = -m_jp1/d_jp1 - 1/2 + c_j/d_jp1
    a_2j   = -m_j  /d_j   + 1/2 + c_j/d_j
    a_2jm1 =  m_j  /d_j   + 1/2 - c_j/d_j
    #
    # ~~~ Assign the computed coefficients to the non-zero positions in the vector a_j
    a_j = torch.zeros_like(x)
    a_j[2*j+2-1] = a_2jp2   # ~~~ a^{(j)}_{2j+2}
    a_j[2*j+1-1] = a_2jp1   # ~~~ a^{(j)}_{2j+1}
    a_j[2*j-1]   = a_2j     # ~~~ a^{(j)}_{2j}
    a_j[2*j-1-1] = a_2jm1   # ~~~ a^{(j)}_{2j-1}
    return a_j

class DualSpline(spline):
    def __init__( self, x, y, eps=1e-6, lr=1e-2 ):
        super().__init__(x,y)
        self.y = y
        self.lamb = torch.randn(self.k-1).to( device=x.device, dtype=x.dtype )**2
        self.a = torch.stack( [ build_a_j(x,j) for j in range(self.k-1) ] )
        self.b = torch.stack( [ build_b_j(x,j) for j in range(self.k-1) ] )
        self.bbt_minus_aat = torch.stack([ torch.outer(self.b[j],self.b[j]) - torch.outer(self.a[j],self.a[j]) for j in range(self.k-1) ])
        self.t = 0  # ~~~ iterations completed thus far of the Frank-Wolfe algorithm
        self.eps = eps
        self.lr = lr
        # self.optimizer = torch.optim.Adam( [self.lamb], lr=lr )
    #
    # ~~~ Solve the dual problem of minimize t^a+MSE(y,z)/m subject to (a[i].T@z)**2 - (b[i].T@z)**2 <= 0 and (z_j-y_j)**2 - t^b \leq 0
    def S_Lemma_1( self, *args, mse_penalty=0, t_squared_objective=False, t_squared_constraint=True, breakpoint_reg=0, **kwargs ):
        if t_squared_constraint and not t_squared_objective: my_warn("Minimizing t subject to |y_j-z_j|^2 \leq t^2 ain't good...")
        aat_minus_bbt = -self.bbt_minus_aat.cpu().numpy()   # ~~~ shape (self.k-1, 2*self.k, 2*self.k)
        m = len(self.y)
        H_o = np.diag(np.concatenate([ (mse_penalty/m)*np.ones(m), [1. if t_squared_objective else 0.] ]))
        c_o = np.concatenate([ -(mse_penalty/m)*self.y.cpu().numpy(), [0. if t_squared_objective else 1/2] ])
        d_o = (mse_penalty/m)*(self.y**2).sum().item()
        H_I = np.concatenate([
                np.pad( aat_minus_bbt, ( (0,0), (0,1), (0,1) ) ),   # ~~~ pad the "actual constraints" with zero for the epigraph variable
                np.stack([ np.diag( j*[0.] + [1.] + (m-j-1)*[0.] + [-1. if t_squared_constraint else 0.] ) for j in range(m)  ])
            ])
        c_I = np.vstack([
                np.zeros(( self.k-1, m+1 )),
                np.hstack(( np.diag(-self.y.cpu().numpy().flatten()), -(not t_squared_constraint)*np.ones((m,1))/2 ))
            ])
        d_I = np.concatenate([
                breakpoint_reg*np.ones(self.k-1),
                self.y.cpu().numpy().flatten()**2
            ])
        problem, lamb, z = solve_dual_of_QCQP( H_o, c_o, d_o, H_I=H_I, c_I=c_I, d_I=d_I, *args, **kwargs )
        # self.lamb = torch.from_numpy(lamb[:(self.k-1)]).to( device=self.lamb.device, dtype=self.lamb.dtype )
        # self.Q = torch.ones_like(self.y).diag() - (self.lamb.reshape(-1,1,1)*self.bbt_minus_aat).sum(dim=0) # ~~~ Q(\lambda) = I - \sum_{j=1}^{k-1} \lambda_j (b_j b_j^T - a_j a_j^T)
        self.z.data = torch.from_numpy(z[:-1])  # ~~~ z = Q(\lambda)^{-1}y 
        return problem
    #
    # ~~~ Solve the dual problem in epigraph form using Simon's suggestion of a linear (rather than non-convex quadratic) epigraph constraint
    def S_Lemma_2( self, *args, mse_penalty=0, t_squared_objective=False, t_squared_constraint=False, breakpoint_reg=0, **kwargs ):
        if t_squared_constraint and not t_squared_objective: my_warn("Minimizing t subject to |y_j-z_j| \leq t^2 ain't good...")
        aat_minus_bbt = -self.bbt_minus_aat.cpu().numpy()   # ~~~ shape (self.k-1, 2*self.k, 2*self.k)
        m = len(self.y)
        H_o = np.diag(np.concatenate([ (mse_penalty/m)*np.ones(m), [1. if t_squared_objective else 0.] ]))
        c_o = np.concatenate([ -(mse_penalty/m)*self.y.cpu().numpy(), [0. if t_squared_objective else 1/2] ])
        d_o = (mse_penalty/m)*(self.y**2).sum().item()
        H_I = np.concatenate([
                np.pad( aat_minus_bbt, ((0,0),(0,1),(0,1)) ),   # ~~~ pad the "actual constraints" with zero for the epigraph variable
                np.zeros(( 2*m, m+1, m+1 ))
            ])
        if t_squared_constraint:
            for j in range(2*m):
                H_I[ k-1+j, -1, -1 ] = -1.
        c_I = np.vstack([
                np.zeros(( self.k-1, m+1 )),
                np.hstack([ np.eye(m)/2, -(not t_squared_constraint)*np.ones((m,1)) ])/2,
                np.hstack([ -np.eye(m)/2, -(not t_squared_constraint)*np.ones((m,1)) ])/2
            ])
        d_I = np.concatenate([
                breakpoint_reg*np.ones(self.k-1),
                -self.y.cpu().numpy(),
                self.y.cpu().numpy()
            ])
        problem, _, z = solve_dual_of_QCQP( H_o, c_o, d_o, H_I=H_I, c_I=c_I, d_I=d_I, *args, **kwargs )
        self.z.data = torch.from_numpy(z[:-1])
        return problem
    #
    # ~~~ Solve a semi-definite relaxation of the dual problem
    def ell_infty_rank_relaxation( self, *args, **kwargs ):
        aat_minus_bbt = -self.bbt_minus_aat.cpu().numpy()   # ~~~ shape (self.k-1, 2*self.k, 2*self.k)
        m = len(self.y)
        H_o = np.zeros(( m+1, m+1 ))
        c_o = np.array( m*[0.] + [1.] )
        d_o = 0.
        H_I = np.concatenate([
                np.pad( aat_minus_bbt, ( (0,0), (0,1), (0,1) ) ),   # ~~~ pad the "actual constraints" with zero for the epigraph variable
                np.stack([ np.diag( j*[0.] + [1.] + (m-j-1)*[0.] + [-1.] ) for j in range(m)  ])
            ])
        c_I = np.vstack([
                np.zeros(( self.k-1, m+1 )),
                np.hstack(( np.diag(-self.y.cpu().numpy().flatten()), np.zeros((m,1)) ))
            ])
        d_I = np.concatenate([
                np.zeros(self.k-1),
                self.y.cpu().numpy().flatten()**2
            ])
        _, X, x = solve_rank_relaxation_of_QCQP( H_o, c_o, d_o, H_I=H_I, c_I=c_I, d_I=d_I, *args, **kwargs )
        self.z.data = torch.from_numpy(x[:-1])
    #
    # ~~~ Minimize MSE(z,y) subject to (a[i].T@z)**2 - (b[i].T@z)**2 + breakpoint_reg <= 0 for all i = 1,...,k-1
    def solve_dual_of_mse_minimization( self, *args, breakpoint_reg=0., **kwargs ):
        aat_minus_bbt = -self.bbt_minus_aat.cpu().numpy()   # ~~~ shape (self.k-1, 2*self.k, 2*self.k)
        m = len(self.y)
        H_o = (1/m)*np.eye(m)
        c_o = -(1/m)*self.y.cpu().numpy()
        d_o = (1/m)*(self.y**2).sum().item()
        problem, _, z = solve_dual_of_QCQP( H_o, c_o, d_o, H_I=aat_minus_bbt, c_I=(self.k-1)*[np.zeros(m)], d_I=(self.k-1)*[breakpoint_reg], *args, **kwargs )
        self.z.data = torch.from_numpy(z)
        return problem
    #
    # ~~~ Generic function that calls the appropriate one of the preceding methods
    def solve_dual_for_z( self, *args, mse_penalty=1, epigraph_constraint=None, epigraph_objective=None, **kwargs ):
        #
        # ~~~ Three constraints are possible
        allowed_epigraph_constraints = ("linear", "quadratic")
        if not epigraph_constraint in allowed_epigraph_constraints:
            raise ValueError(f"epigraph_constraint must be one of {allowed_epigraph_constraints}")
        #
        # ~~~ Three objectives are possible
        allowed_epigraph_objectives = ("linear", "quadratic")
        if not epigraph_objective in allowed_epigraph_objectives:
            raise ValueError(f"epigraph_objectives must be one of {allowed_epigraph_objectives}")
        #
        # ~~~ Solve the dual of max norm minimization in epigraph form, possibly also penalizing MSE in the objective function
        if epigraph_constraint == "linear":
            self.S_Lemma_2( *args, mse_penalty=mse_penalty, quadratic_objective=(epigraph_objective=="quadratic") **kwargs )
        else:
            self.S_Lemma_1( *args, mse_penalty=mse_penalty, quadratic_objective=(epigraph_objective=="quadratic") **kwargs )
        #
        # ~~~ Diagnostics
        worst_offender = self.compute_violation().min().item()
        if worst_offender<1:
            my_warn(f"The approximate primal solution fails to define an element of V (error is roughly {1-worst_offender}). If a primal solution is desired, try specifying a slightly larger (but still very small) value for the argument `breakpoint_reg` (default is 0)")
    #
    # ~~~ Minimize the constant function 1 subject to \|z-y\|_2^2<=noise and (a[i].T@z)**2 - (b[i].T@z)**2 <= 0
    def S_Lemma_4( self, noise=0.1, *args, **kwargs  ):
        #
        # ~~~ Set the objective function to be identically equal to 1.
        m = len(self.y)
        d_o = 1.
        c_o = np.zeros(m)
        H_o = np.eye(m)
        #
        # ~~~ Set the inequality constraints
        aat_minus_bbt = -self.bbt_minus_aat.cpu().numpy()                   # ~~~ shape (self.k-1, m, m)
        H_I = np.concatenate([ aat_minus_bbt, np.eye(m)[np.newaxis,:]/m ])  # ~~~ shape ( self.k,  m, m)
        c_I = (self.k-1)*[np.zeros(m)] + [-self.y.cpu().numpy()/m]
        d_I = (self.k-1)*[0.] + [ (self.y.cpu()**2).mean().numpy() - noise ]
        #
        # ~~~ Solve the dual
        _, _, z = solve_dual_of_QCQP( H_o, c_o, d_o, H_I=H_I, c_I=c_I, d_I=d_I, *args, **kwargs )
        #
        # ~~~ Process the results
        # self.lamb = torch.from_numpy(lamb).to( device=self.lamb.device, dtype=self.lamb.dtype )
        # self.Q = torch.from_numpy( (lamb.reshape(-1,1,1)*H_I).sum(axis=0) ).to( device=self.lamb.device, dtype=self.lamb.dtype )
        # self.z.data = torch.linalg.solve( self.Q, self.y*lamb[-1]/m )
        # print(f"The dual max is {gamma}")
        self.z.data = torch.from_numpy(z)
    #
    # ~~~ Minimize t+mse_penalty*MSE(y,z) subject to |s_jx + c_j - y| \leq t for both data pairs (x,y), for j=1,...,k, and subject to p_j(s_1,...,s_k,c_1,...,c_k) \leq 0
    def D_kappa( self, *args, M=0, kappa=0, mse_penalty=0, quadratic_objective=True, announce_eigenvalues=True, **kwargs ):
        #
        # ~~~ Define objects of the correct size
        k = self.k
        m = len(self.y)
        H_o = np.diag(np.concatenate([ (mse_penalty/m)*np.ones(m), [1. if quadratic_objective else 0.] ])) #np.zeros((m+1,m+1))
        c_o = np.concatenate([ -(mse_penalty/m)*self.y.cpu().numpy(), [0. if quadratic_objective else 1/2] ]) #np.array( 2*k*[0.] + [1/2] )
        d_o = (mse_penalty/m)*(self.y**2).sum().item() #0
        H_I = np.zeros(( k-1+2*m, 2*k+1, 2*k+1 ))
        c_I = np.zeros(( k-1+2*m, 2*k+1 ))
        d_I = np.zeros( k-1+2*m )
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
        # ~~~ Build the "actual constraints"
        for j in range(k-1):
            j += 1  # ~~~ use 1-based indexing j=1,...,k-1
            delta_over_2 = (self.x[(2*j)-1] - self.x[(2*j-1)-1]).item()/2           # ~~~ == (x_{2j} - x_{2j-1})/2
            shifted_midpoint = (self.x[(2*j+1)-1] + self.x[(2*j)-1]).item()/2 + M   # ~~~ == (x_{2j+1} + x_{2j})/2 + M
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
            # penalty_coefficient = kappa/M if divide_by_M else kappa/abs(shifted_midpoint)
            # H_I[ j, :, : ] += penalty_coefficient*np.eye(2*k+1)
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
                c_I[ k-1+index_2j_minus_i, j   ] = (evaluation_site+M)
                c_I[ k-1+index_2j_minus_i, k+j ] = 1
                c_I[ k-1+index_2j_minus_i, -1  ] = -1
                d_I[ k-1+index_2j_minus_i ] = -training_label
                #
                # ~~~ Add the constraint -s_j*(x+M) - c_j + y - t \leq 0
                c_I[ k-1+m+index_2j_minus_i, j   ] = -(evaluation_site+M)
                c_I[ k-1+m+index_2j_minus_i, k+j ] = -1
                c_I[ k-1+m+index_2j_minus_i, -1  ] = -1
                d_I[ k-1+m+index_2j_minus_i ] = training_label
        #
        # ~~~ Solve the dual
        problem, _, s_c_t = solve_dual_of_QCQP( H_o, c_o, d_o, H_I=H_I, c_I=c_I/2, d_I=d_I, *args, **kwargs )
        s, c, t = np.array_split( s_c_t, [k,2*k] )
        for j in range(k):
            for i in [1,0]:
                j += 1  # ~~~ use 1-based indexing j=1,...,k
                appropriate_index = (2*j-i)-1
                j -= 1  # ~~~ return to 0-based indexing
                with torch.no_grad(): self.z.data[appropriate_index] = s[j]*(self.x[appropriate_index]+M) + c[j]
        print("")
        print(f"t: {t}")
        print(f"objective: {problem.objective.value}")
        print("")
        return problem
    #
    # ~~~ Minimize t+mse_penalty*MSE(y,z) subject to |s_jx + c_j - y| \leq t for both data pairs (x,y), for j=1,...,k, and subject to p_j(s_1,...,c_k) + \kappa\|s_1,...,s_k\|^2 \leq 0 which is equivalent to (s_1,...,c_k) lying in the eigenspace of p_j's Hessian's smallest eigenvalue, which we enforce as a linear equality constraint
    def P_kappa_1( self, *args, M=0, mse_penalty=0, quadratic_objective=False, announce_eigenvalues=True, **kwargs ):
        #
        # ~~~ Define objects of the correct size
        k = self.k
        s = cp.Variable(k)
        c = cp.Variable(k)
        t = cp.Variable()
        alpha = cp.Variable(k-1)
        epigraph_objective = t**2 if quadratic_objective else t
        if mse_penalty>0: epigraph_objective = epigraph_objective + mse_penalty*cp.sum_squares( self.y.cpu().numpy() - self.z.detach().cpu().numpy() )
        objective = cp.Minimize( epigraph_objective )
        constraints = [ t>=0 ]
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
                #
                # ~~~ Add the constraint s_j*(x+M) + c_j - y - t \leq 0
                constraints.append( s[j]*(evaluation_site+M) + c[j] - training_label - t <= 0 )
                #
                # ~~~ Add the constraint -s_j*(x+M) - c_j + y - t \leq 0                
                constraints.append( -s[j]*(evaluation_site+M) - c[j] + training_label - t <= 0 )
        #
        # ~~~ Solve it
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
    v.z.data = torch.randn(m)
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
    if N is None:
        val = v.D_kappa( M=2, kappa=1, quadratic_objective=True, mse_penalty=0 )
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
    fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), show=False, title="The Result of Some Quadratic Program" )
    with torch.no_grad():
        nodes = v.compute_break_points()
        ax.scatter( nodes, v(nodes), color="blue", alpha=0.4 )
    plt.show()
    my_s = np.zeros(k)
    my_c = np.zeros(k)
    for j in range(k):
        my_s[j] = (v.z[2*(j+1)-2] - v.z[2*(j+1)-1]) / (x_train[2*(j+1)-2] - x_train[2*(j+1)-1])
        my_c[j] = v.z[2*(j+1)-1] - my_s[j]*x_train[2*(j+1)-1]

if False:
    #
    # ~~~ Try taking that as the initialization for a ReLU network
    v = CarefulNet(x_train)
    with torch.no_grad():
        slopes = (best_z[1::2] - best_z[::2]) / (x_train[1::2] - x_train[::2])
        v.relu_net[0].bias.data = -nodes.reshape(v.relu_net[0].bias.data.shape)
        v.a.data = slopes[0]
        v.relu_net[-1].weight.data = slopes.diff().reshape(v.relu_net[-1].weight.data.shape)
        v.relu_net[-1].bias.fill_( best_z[0] - slopes[0]*x_train[0] )
    fig, ax = points_with_curves( x=x_train, y=y_train, grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(v,f), show=False, title="MSE Minimization Subject to Constraints on the Location of Breakpoints" )
    with torch.no_grad():
        ax.scatter( nodes, v(nodes.reshape(-1,1)), color="blue", alpha=0.4 )
    plt.show()
    #
    # ~~~ Train it
    optimizer = torch.optim.Adam( v.parameters(), lr=1e-2 )
    x_train_vertical = x_train.reshape(-1,1)
    gif = GifMaker()
    fig, ax = points_with_curves( x=x_train, y=y_train, grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(v,f), title="MSE Minimization Subject to Constraints on the Location of Breakpoints", show=False )  
    gif.capture()
    with support_for_progress_bars():
        for _ in trange(10000):
            predictinons = v(x_train_vertical)
            max_error = (y_train-predictinons).abs().max()
            max_error.backward()
            optimizer.step()
            optimizer.zero_grad()
            v.project()
            if (_+1)%100:
                fig, ax = points_with_curves( x=x_train, y=y_train, grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(v,f), title="MSE Minimization Subject to Constraints on the Location of Breakpoints", show=False, fig=fig, ax=ax )
                gif.capture()
        points_with_curves( x=x_train, y=y_train, grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(v,f), title="MSE Minimization Subject to Constraints on the Location of Breakpoints", fig=fig, ax=ax )
        gif.develop()
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
    ocassional_model = nn.Sequential(
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
    ocassional_optimizer = torch.optim.Adam( ocassional_model.parameters(), lr=1e-2 )
    with support_for_progress_bars():
        for epoch in trange(10000):
            loss = (( model(x_train).flatten() - y_train )**2).mean()
            loss.backward()
            big_loss = (( big_model(x_train).flatten() - y_train )**2).mean()
            big_loss.backward()
            if epoch % 10 == 0:
                ocassional_loss = (( ocassional_model(x_train).flatten() - y_train )**2).mean()
                ocassional_loss.backward()
            for opt in ( optimizer, big_optimizer, ocassional_optimizer ):
                opt.step()
                opt.zero_grad()
    fig, ax = points_with_curves(
            x = x_train,
            y = y_train,
            curves = ( big_model, ocassional_model, model, v, f ),
            curve_labels = ( "Large Network Trained with ADAM", "Large Network Trained with ADAM and Early Stopping", "Same Small Network Trained with ADAM", "Small Network Trained with Our Method", "Ground Truth" ),
            curve_colors = ( "black", "grey", "red", "blue", "green"),
            curve_marks  = [ "-", "-", "-", "-", "--" ],
            show = False,
            title = "Comparison of Our Model with ADAM and Larger Neural Networks",
            model_fit = False
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
    plt.legend( legend_handles, legend_labels, fontsize=17 )
    plt.show()
    print(v.compute_violation())

#