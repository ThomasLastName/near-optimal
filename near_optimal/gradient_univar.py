
import torch
from torch import nn
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from quality_of_life.my_plt_utils import points_with_curves
from quality_of_life.my_base_utils import support_for_progress_bars

class spline(nn.Module):
    def __init__( self, x ):
        super().__init__()
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
        self.z = nn.Parameter( torch.randn_like(x) )
    #
    # ~~~ Compute the thing that we want to be non-negative
    def compute_violation( self ):
        a = torch.zeros_like(self.d)
        s = torch.zeros_like(self.d)
        for j in range(self.k):
            j += 1
            a[j-1] = (self.z[2*j-1] + self.z[2*j-1-1]) / 2              # ~~~ (z_{2j} + z_{2j-1}) / 2
            s[j-1] = (self.z[2*j-1] - self.z[2*j-1-1]) / self.d[j-1]    # ~~~ (z_{2j} - z_{2j-1}) / (x_{2j} - x_{2j-1})
        # for j in range(self.k-1):
        #     foo[j] = s[j+1]*self.m[j+1] - s[j]*self.m[j] - (a[j+1] - a[j]) - second_derivative[j]*self.c[j] # ~~~ "x" + (s_{j+1}-x_j)*c_j
        second_derivative = s.diff()    # ~~~ j-th component is (s[j+1] - s[j])
        violator = (s*self.m).diff() - a.diff() - second_derivative*self.c
        return self.D/2 * second_derivative.abs() - violator.abs()
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
    # ~~~ Compute v(x)
    def forward( self, x ):
        nodes = self.compute_break_points()
        indices = torch.searchsorted( nodes, x )
        a = torch.zeros_like(self.d)
        s = torch.zeros_like(self.d)
        for j in range(self.k):
            j += 1
            a[j-1] = (self.z[2*j-1] + self.z[2*j-1-1]) / 2              # ~~~ (z_{2j} + z_{2j-1}) / 2
            s[j-1] = (self.z[2*j-1] - self.z[2*j-1-1]) / self.d[j-1]    # ~~~ (z_{2j} - z_{2j-1}) / (x_{2j} - x_{2j-1})
        return a[indices] + s[indices] * (x - self.m[indices])

torch.manual_seed(2024)
k = 15
m =  2*k
f = lambda x: torch.sin(2*torch.pi*x)
x_train = torch.linspace(-1,1,m)
y_train = f(x_train)
v = spline(x_train)
x_test = torch.linspace(-1,1,1001)
y_test = f(x_test)
# points_with_curves( x=x_train,  y=y_train, curves=(v,f) )

penalty_coefficient = .5
penalty_fn = lambda x: torch.clamp(-x,min=0).max()  # ~~~ returns zero if x\geq0, else returns something positive
lr = 1e-2
N = 1000
optimizer = torch.optim.Adam( v.parameters(), lr=lr )
history = []
pbar = tqdm( desc="Using Gradient Descent", total=N, ascii=' >=' )
with support_for_progress_bars():
    for _ in range(N):
        should_be_nonnegative = v.compute_violation()
        max_error = (y_train-v.z).abs().max()
        loss = max_error + penalty_coefficient*penalty_fn(should_be_nonnegative)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _ = pbar.update()
        history.append(max_error.item())
        pbar.set_postfix({ "max_error" : f"{history[-1]:<4.4f}" })

pbar.close()

fig, ax = points_with_curves( x=x_train,  y=y_train, curves=(v,f), show=False )
with torch.no_grad():
    nodes = v.compute_break_points()
    ax.scatter( nodes, v(nodes) )
    plt.show()

print(should_be_nonnegative)
