
import torch
from torch import nn
from near_optimal.gradient_univar import spline
import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from quality_of_life.my_plt_utils import points_with_curves
from quality_of_life.my_base_utils import support_for_progress_bars



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
    # ~~~ Assign the computed coefficients to the non-zero positions in the vector a_j
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

class dual_spline(spline):
    def __init__( self, x, y, eps=1e-6, lr=1e-2 ):
        super().__init__(x,y)
        self.y = y
        self.lamb = torch.randn(k-1).to( device=x.device, dtype=x.dtype )**2
        self.a = torch.stack( [ build_a_j(x,j) for j in range(k-1) ] )
        self.b = torch.stack( [ build_b_j(x,j) for j in range(k-1) ] )
        self.bbt_minus_aat = torch.stack([ torch.outer(self.b[j],self.b[j]) - torch.outer(self.a[j],self.a[j]) for j in range(k-1) ])
        self.t = 0  # ~~~ iterations completed thus far of the Frank-Wolfe algorithm
        self.eps = eps
        self.lr = lr
        # self.optimizer = torch.optim.Adam( [self.lamb], lr=lr )
    #
    # ~~~ 
    def PGD_step(self):
        #
        # ~~~ Compute the gradient of F(\lambda)
        self.Q = torch.ones_like(self.y).diag() - (self.lamb.reshape(-1,1,1)*self.bbt_minus_aat).sum(dim=0) # ~~~ Q(\lambda) = I - \sum_{j=1}^{k-1} \lambda_j (b_j b_j^T - a_j a_j^T)
        z = torch.linalg.solve( self.Q, self.y )            # ~~~ z = Q(\lambda)^{-1}y 
        g = ((self.a@z)**2 - (self.b@z)**2).cpu().numpy()   # ~~~ \grad_\lambda F(\lambda)
        objective_before_update = -torch.inner( self.z, self.y )
        self.z.data = z
        self.lamb += self.lr*g
        # print( self.lamb, end="\n" )
        #
        # ~~~ Project onto the constraint set
        s = cp.Variable(self.k-1)
        objective = cp.Minimize( cp.sum_squares( s - self.lamb.double().cpu().numpy() ))
        bbt_minus_aat = self.bbt_minus_aat.cpu().numpy()
        R = sum(s[i] * bbt_minus_aat[i] for i in range(14))
        constraints = [
                s >= 0,
                R << (1-self.eps)*np.eye(2*self.k)
            ]
        problem = cp.Problem(objective, constraints)
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
        objective_before_update = -torch.inner( self.z, self.y )
        g *= self.lr
        self.z.data = z
        #
        # ~~~ Solve the Frank-Wolfe subproblem to find a better update direction than the gradient
        s = cp.Variable(self.k-1)
        objective = cp.Maximize( g@s )
        bbt_minus_aat = self.bbt_minus_aat.cpu().numpy()
        R = sum(s[i] * bbt_minus_aat[i] for i in range(14))
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
            self.lr *= (1+0.1)
            alpha = 2/(self.t+2)
            self.lamb = (1-alpha)*self.lamb + alpha*torch.from_numpy(s.value).to( device=self.lamb.device, dtype=self.lamb.dtype )
            self.t += 1
        return objective_before_update, duality_gap
    #
    # ~~~
    # def forward(self, x):
    #     #
    #     # ~~~ Compute the dual solution
    #     self.Q = torch.ones_like(self.y).diag() - (self.lamb.reshape(-1,1,1)*self.bbt_minus_aat).sum(dim=0) # ~~~ Q(\lambda) = I - \sum_{j=1}^{k-1} \lambda_j (b_j b_j^T - a_j a_j^T)
    #     self.z = torch.linalg.solve( self.Q, self.y )            # ~~~ z = Q(\lambda)^{-1}y
    #     #
    #     # ~~~ Compute the primal solution
    #     # return

if __name__ == "__main__":
    #
    # ~~~ Config
    torch.manual_seed(2024)
    torch.set_default_dtype(torch.double)
    k = 15
    m =  2*k
    f = lambda x: torch.sin(2*torch.pi*x)
    x_train = torch.linspace(-1,1,m)
    y_train = f(x_train)
    v = dual_spline( x_train, y_train )
    x_test = torch.linspace(-1,1,1001)
    y_test = f(x_test)
    # points_with_curves( x=x_train,  y=y_train, curves=(v,f) )
    N = 100
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
    fig, ax = points_with_curves( x=x_train, y=y_train, curves=(v,f), show=False, title="MSE Minimization with Hard Constraints" )
    with torch.no_grad():
        nodes = v.compute_break_points()
        ax.scatter( nodes, v(nodes), color="blue", alpha=0.4 )
        plt.show()
    print(v.compute_violation())
