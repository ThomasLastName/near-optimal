"""
Here, we test a couple of things simultaneously.

First, we test one of the *many* different equivalent formulations of
the constraints on break points: this formulation proposed by Simon.

Second, we test constraining error(z) \leq noise where noise not a variable, but rather
rather user-specified level of how much noise would be tolerable. Specifically,
 - We apply ADAM with a projection step to the problem
    "minimize break-point-constraints subject to \max_j |z_j-y_j| \leq noise"
    where the projection of z onto this box constraint is quite easy to implement.
 - We solve the dual of the quadratic feasibility program
    "minimize break-point-constraints subject to mse(z,y) \leq noise"

These both work okay-ish but not, like, spectacularly good or anything.
No obvious improvement over the other methods implemented in `quadratic_univar.py`
"""

import torch
from near_optimal.penalty_functions_NOT_RECOMMENDED import spline
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from quality_of_life.my_plt_utils import points_with_curves, GifMaker
from quality_of_life.my_base_utils import support_for_progress_bars
from quality_of_life.my_torch_utils import hot_1_encode_an_integer
from quality_of_life.my_cvx_utils import solve_dual_of_QCQP


### ~~~
## ~~~ Compute the vector c = c_j(x_new) such that v_j(x_new) = \langle c,z \rangle
### ~~~

def build_c_j( x_train, j, x_new ):
    #
    # ~~~ Use 1-indexing, assuming that the given j is in zero-indexing to begin with
    x = x_train
    assert len(x)%2==0
    k = len(x)//2
    assert j >= 0 and j <= k-1
    j += 1
    x_2j   =  x[2*j-1]
    x_2jm1 =  x[2*j-1-1]
    d_j    =  (x_2j - x_2jm1)  # ~~~ d_j = x_{2j} - x_{2j-1}, the length of one of the intervals in which no break point is allowed
    m_j    =  (x_2jm1 + x_2j)/2
    #
    # ~~~ Compute the non-zero coordinates of the vector c_j(x_new)
    c_2j   =  1/2 + x_new/d_j - m_j/d_j
    c_2jm1 =  1/2 + m_j/d_j - x_new/d_j
    #
    # ~~~ Assign the computed coefficients to the non-zero positions in the vector c_j
    c_j = torch.zeros_like(x)
    c_j[2*j-1]   = c_2j
    c_j[2*j-1-1] = c_2jm1
    return c_j

def build_simons_a_j_and_b_j(x,j):
    #
    # ~~~ Use 1-indexing, assuming that the given j is in zero-indexing to begin with
    assert len(x)%2==0
    k = len(x)//2
    assert j >= 0 and j <= k-1
    j += 1
    x_2jp1 =  x[2*j+1-1]    # ~~~ x_{2*j+1}
    x_2j   =  x[2*j-1]      # ~~~ x_{2*j}
    make_basis_vector = hot_1_encode_an_integer(len(x))
    e_2jp1 =  make_basis_vector(2*j+1-1)
    e_2j   =  make_basis_vector(2*j-1)
    #
    # ~~~ Convert back to zero-indexing, which build_c_j expects
    j -= 1
    a_j    =  build_c_j( x, j, x_2jp1 ) - e_2jp1  # ~~~ v_j(x_{2j+1}) - z_{2j+1} = a^T@z
    b_j    =  build_c_j( x, j+1, x_2j ) - e_2j    # ~~~ v_{j+1}(x_{2j}) - z_{2j} = b^T@z
    return a_j, b_j


class semidefinite_spline(spline):
    def __init__( self, x, y, eps=1e-5, delta=1e-5 ):
        super().__init__(x,y)
        a = []
        b = []
        abt = []
        k = len(x)//2
        for j in range(k-1):
            a_j, b_j = build_simons_a_j_and_b_j(x,j)
            a.append(a_j)
            b.append(b_j)
            abt.append( torch.outer(a_j,b_j) )
        self.a = torch.stack(a)
        self.b = torch.stack(b)
        self.abt = torch.stack(abt)
        self.y = y
        self.lamb = torch.randn(k-1).to( device=x.device, dtype=x.dtype )**2
    #
    # ~~~ Solve the dual problem in epigraph form
    def S_Lemma( self, *args, **kwargs ):
        abt = self.abt.cpu().numpy()   # ~~~ shape (self.k-1, 2*self.k, 2*self.k)
        m = len(self.y)
        H_o = np.eye(m)/m
        c_o = -self.y.cpu().numpy()/m
        d_o = (self.y**2).sum().item()/m
        _, Lagrange_multiplers, z = solve_dual_of_QCQP( H_o, c_o, d_o, H_I=-abt, c_I=(self.k-1)*[np.zeros(m)], d_I=(self.k-1)*[0.], *args, **kwargs )
        self.z.data = torch.from_numpy(z)
    #
    # ~~~ Solve (in epigraph form) the dual of the quadratic feasibility problem of the feasibility problem ||z-y||_2^2/m\leq\noise and |a[i].T@z|\leq|b[i].T^@z| for all i
    def noisy_S_Lemma( self, noise=0.1, reg=1, *args, **kwargs  ):
        #
        # ~~~ Set the objective function to be identically zero
        m = len(self.y)
        H_o = np.eye(m)/m   # ~~~ since this is a feasibility program, it doesn't matter what we minimize; so we might as well minimize a strongly convex function which we hope will be small, anyway: the mean squared error
        c_o = -self.y.cpu().numpy()/m
        d_o = (self.y.cpu()**2).mean().numpy()
        #
        # ~~~ Set the inequality constraints
        abt = self.abt.cpu().numpy()                                # ~~~ shape ( self.k-1, m, m )
        H_I = np.concatenate([ -abt, np.eye(m)[np.newaxis,:] ])   # ~~~ shape ( self.k, m, m )
        c_I = (self.k-1)*[np.zeros(m)] + [-self.y.cpu().numpy()]  # ~~~ shape ( self.k, m )
        d_I = (self.k-1)*[0.] + [ (self.y.cpu()**2).sum().numpy() - m*noise ]  # ( self.k, )
        #
        # ~~~ Solve the dual
        _, Lagrange_multiplers, z = solve_dual_of_QCQP( H_o, c_o, d_o, H_I=H_I, c_I=c_I, d_I=d_I, *args, **kwargs )
        self.z.data = torch.from_numpy(z)
    #
    # ~~~ Project z onto the set of z's that satisfy \|z-y\|_{\ell^\infty}\leq\eta
    def ell_infty_projection( self, eta=0.1 ):
        with torch.no_grad():
            self.z.clamp_( min=self.y-eta, max=self.y+eta )


if __name__ == "__main__":
    #
    # ~~~ Config
    from near_optimal.quadratic_univar import x_train, y_train, f
    v = semidefinite_spline( x_train, y_train )
    noise = 0.05
    if noise and noise>0:
        #
        # ~~~ Try gradient descent on the problem \min_z \max_\ell -\langle z,a_\ell \rangle*\langle z,b_\ell \rangle subject to \|z-y\|_\infty \leq \eta
        v.ell_infty_projection(eta=noise)
        optimizer = torch.optim.Adam( v.parameters(), lr=1e-3 )
        gif = GifMaker()
        fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), title="Minimize the Violation Subject to an \ell^\infty Constraint", show=False )
        gif.capture()
        with support_for_progress_bars():
            for _ in tqdm(range(200)):
                predictions = v.z   # == v(x_train)
                loss = (-(v.a@predictions) * (v.b@predictions)).max()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                v.ell_infty_projection(eta=noise)
                if (_+1)%10==0:
                    fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), title="Minimize the Violation Subject to an \ell^\infty Constraint", show=False, fig=fig, ax=ax )
                    gif.capture()
            fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), title="Minimize the Violation Subject to an \ell^\infty Constraint", fig=fig, ax=ax, show=False )
            gif.develop()
        print("")
        if v.compute_violation().detach().min().item()>=1: print(f"Constraints are satisfied and we achieved a max abs error of {max(abs(v.z-y_train)).item()}. We know from the other experiments that it must be > 0.048.")
        else: print("Constraints not satisfied")
        print("")
    #
    # ~~~ Solve the dual program (THIS ONE DOESN'T WORK AS WELL, not even if you adjust the dual solution as described in the paper)
    v.S_Lemma( eps_abs=1e-2, eps_rel=1e-2, eps_infeas=1e-2 ) if noise is None else v.noisy_S_Lemma( noise, eps_abs=1e-2, eps_rel=1e-2, eps_infeas=1e-2  )
    best_z = v.z.data.clone()
    v.z.data = best_z
    from near_optimal.quadratic_univar import DualSpline
    v = DualSpline( x_train, v(x_train).detach() )  # ~~~ __init__ method adjusts the dual solution as described in the paper
    fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), show=False, title="Via dual programming" )
    with torch.no_grad():
        nodes = v.compute_break_points()
        ax.scatter( nodes, v(nodes), color="blue", alpha=0.4 )
    plt.show()
