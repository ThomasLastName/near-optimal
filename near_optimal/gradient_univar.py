
import torch
from torch import nn
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from quality_of_life.my_plt_utils import points_with_curves
from quality_of_life.my_base_utils import support_for_progress_bars

class spline(nn.Module):
    def __init__( self, x, y=None ):
        super().__init__()
        if not isinstance( x, torch.Tensor ):                   x = torch.tensor(x)
        if y is not None and not isinstance( y, torch.Tensor ): y = torch.tensor(y)
        x = x.squeeze()
        assert x.dim()==1
        x = x.sort().values
        m = len(x)
        assert m%2==0
        k = m//2
        m = torch.zeros(k).to( device=x.device, dtype=x.dtype )
        d = torch.zeros(k).to( device=x.device, dtype=x.dtype )
        c = torch.zeros(k-1).to( device=x.device, dtype=x.dtype )
        D = torch.zeros(k-1).to( device=x.device, dtype=x.dtype )
        for j in range(k):
            j += 1                                  # ~~~ guys, I fixed zero indexing
            m[j-1] = (x[2*j-1] + x[2*j-1-1]) / 2    # ~~~ midpoint (x_{2j} + x_{2j-1})/2 of the interval where no break points are allowed
            d[j-1] = x[2*j-1] - x[2*j-1-1]          # ~~~ length \delta_j = x_{2j} - x_{2j-1} of the interval where no break points are allowed
        for j in range(k-1):
            j += 1                                  # ~~~ guys, I fixed zero indexing
            c[j-1] = (x[2*j-1] + x[2*j+1-1]) / 2    # ~~~ midpoint (x_{2j} + x_{2j+1})/2 of the inverval where one break point is allowed
            D[j-1] = (x[2*j+1-1] - x[2*j-1])        # ~~~ length of the inverval where one break point is allowed
        assert d.min()>0 and D.min()>0
        self.x = x
        self.k = k
        self.m = m
        self.d = d
        self.c = c
        self.D = D
        self.z = nn.Parameter( torch.randn_like(x) if y is None else torch.clone(y) )   # ~~~ note: I think of self.z as being self(self.x), but mathematically, that's only true when the constraint is satisfied
        self.compute_slopes_and_intercepts()
    #
    # ~~~ Compute the thing that we want to be \geq 1
    def compute_violation( self ):
        a = torch.zeros_like(self.d)
        s = torch.zeros_like(self.d)
        for j in range(self.k):
            j += 1
            a[j-1] = (self.z[2*j-1] + self.z[2*j-1-1]) / 2              # ~~~ (z_{2j} + z_{2j-1}) / 2
            s[j-1] = (self.z[2*j-1] - self.z[2*j-1-1]) / self.d[j-1]    # ~~~ (z_{2j} - z_{2j-1}) / (x_{2j} - x_{2j-1})
        second_deriv = s.diff()                                         # ~~~ j-th component is (s[j+1] - s[j])
        violator = (s*self.m).diff() - a.diff() - second_deriv*self.c   # ~~~ constraint is that this should have absolute value at most self.D/2*second_deriv.abs()
        return self.D/2 * second_deriv.abs() / violator.abs()
    #
    # ~~~ Compute the points at which the neighboring lines intersect
    def compute_break_points( self ):
        a = torch.zeros_like(self.d)
        s = torch.zeros_like(self.d)
        nodes = torch.zeros_like(self.c)
        for j in range(self.k):
            j += 1
            a[j-1] = (self.z[2*j-1] + self.z[2*j-1-1]) / 2              # ~~~ (z_{2j} + z_{2j-1}) / 2
            s[j-1] = (self.z[2*j-1] - self.z[2*j-1-1]) / self.d[j-1]    # ~~~ (z_{2j} - z_{2j-1}) / (x_{2j} - x_{2j-1})
        for j in range(self.k-1):
            nodes[j] = (
                    s[j+1]*self.m[j+1] - s[j]*self.m[j] - (a[j+1] - a[j])
                ) / (
                    s[j+1] - s[j]
                )
        self.nodes = nodes
        return nodes
    #
    # ~~~ Using z, compute the slopes and intercepts of each linear piece
    def compute_slopes_and_intercepts(self):
        c = torch.zeros_like(self.d)
        s = torch.zeros_like(self.d)
        for j in range(self.k):
            j += 1
            s[j-1] = (self.z[2*j-1] - self.z[2*j-1-1]) / self.d[j-1]    # ~~~ (z_{2j} - z_{2j-1}) / (x_{2j} - x_{2j-1})
            c[j-1] = self.z[2*j-1] - s[j-1]*self.x[2*j-1]
        self.slopes = s
        self.intercepts = c
    #
    # ~~~ Compute v(x)
    def forward( self, x ):
        nodes = self.compute_break_points()
        indices = torch.searchsorted( nodes, x )
        c = torch.zeros_like(self.d)
        s = torch.zeros_like(self.d)
        for j in range(self.k):
            j += 1
            s[j-1] = (self.z[2*j-1] - self.z[2*j-1-1]) / self.d[j-1]    # ~~~ (z_{2j} - z_{2j-1}) / (x_{2j} - x_{2j-1})
            c[j-1] = self.z[2*j-1] - s[j-1]*self.x[2*j-1]
        return s[indices]*x + c[indices]

if __name__ == "__main__":
    #
    # ~~~ Config
    torch.manual_seed(2025)
    k = 15
    m =  2*k
    f = lambda x: torch.sin(2*torch.pi*x)
    scale = 0.2
    x_train = torch.linspace(-1,1,m)
    y_train = f(x_train) + scale*torch.randn_like(x_train)
    v = spline( x_train, y_train )
    x_test = torch.linspace(-1,1,1001)
    y_test = f(x_test)
    # points_with_curves( x=x_train,  y=y_train, curves=(v,f) )
    penalty_coefficient = 0.5
    penalty_fn = lambda x: torch.clamp(1-x,min=0).max()**2  # ~~~ returns zero if x\geq1, else returns something positive
    lr = 1e-3
    N = 2000
    optimizer = torch.optim.Adam( v.parameters(), lr=lr )
    #
    # ~~~ Gradient Descent
    history = []
    with support_for_progress_bars():
        pbar = tqdm( desc="Using Gradient Descent", total=N, ascii=' >=' )
        for _ in range(N):
            should_be_geq_1 = v.compute_violation()
            max_error = (y_train-v.z).abs().max()
            loss = max_error + penalty_coefficient*penalty_fn(should_be_geq_1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            _ = pbar.update()
            history.append(max_error.item())
            pbar.set_postfix({ "max_error" : f"{history[-1]:<4.4f}" })
    pbar.close()
    fig, ax = points_with_curves( x=x_train,  y=y_train, curves=(v,f), show=False, title=r"$\ell^\infty$ Error Minimization with Soft Constraints (Penalty Functions)"  )
    with torch.no_grad():
        nodes = v.compute_break_points()
        ax.scatter( nodes, v(nodes), color="blue", alpha=0.4 )
        plt.show()
    print("")
    print("Each constraint is satisfied *iff* the corresponding value is \geq 1:")
    print("")
    print(v.compute_violation())
