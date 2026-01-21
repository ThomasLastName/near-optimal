
import torch
from matplotlib import pyplot as plt
from tqdm import trange
from quality_of_life.my_plt_utils import points_with_curves
from quality_of_life.my_base_utils import support_for_progress_bars 

from near_optimal.quadratic_univar import DualSpline
from near_optimal.PGD_univar import RigorousNet

# #
# # ~~~ Config
# torch.manual_seed(2025)
# k = 5
# m =  2*k
# f = lambda x: torch.sin(2*torch.pi*x)
# noise_level = 0
# scale = 0.1
# x_train = torch.linspace(-1,1,m).reshape(-1,1)
# y_train = f(x_train) + scale*torch.randn_like(x_train)
# x_test = torch.linspace(-1,1,1001)
# y_test = f(x_test)
# model = RigorousNet(x_train)
# lr = 1e-2
# N = 22000
# optimizer = torch.optim.Adam( model.parameters(), lr=lr )
# with support_for_progress_bars():
#     for _ in trange(N):
#         predictinons = model(x_train)
#         max_error = (y_train-predictinons).abs().max()
#         max_error.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         model.project()

# points_with_curves( x=x_train.squeeze(), y=y_train.squeeze(), grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(model,f), title=r"$\ell^\infty$ Error Minimization with Hard Constraints" )

# y_train = model(x_train).squeeze().detach()
# x_train = x_train.squeeze()
# v = DualSpline( x_train, y_train )
# v.D_kappa()
# points_with_curves( x=x_train, y=y_train, curves=(v,f), title=r"$\ell^2$ Error Minimization with Soft Constraints" )

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
v = DualSpline( x_train, y_train )
M_max = 3
M_n = 4

D_kappa_0_curve = []
D_kappa_1_curve = []
for M in torch.linspace( 0, M_max, M_n ):
    for kappa in (0,1):
        for mse_penalty in (0,1):
            print( v.D_kappa( M=M, kappa=kappa, mse_penalty=mse_penalty, solver="SCS" ) )