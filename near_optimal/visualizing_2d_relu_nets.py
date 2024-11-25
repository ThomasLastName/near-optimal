
import warnings
from tqdm import trange
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from quality_of_life.my_torch_utils import cartesian_product
from quality_of_life.my_plotly_utils import cell_viz
from quality_of_life.my_plt_utils import GifMaker, close_all_figures
from quality_of_life.my_base_utils import support_for_progress_bars
from near_optimal.compute_discontinuity_segments import minimalist_heatmap_where_relu_net_is_not_smooth


warnings.filterwarnings(
    "ignore", 
    message  = "Solution may be inaccurate. Try another solver.*", 
    category = UserWarning, 
    module   = r".*cvxpy.*"
)

torch.set_default_dtype(torch.double)
COMPUTE_SEGMENTS = True
HOW_OFTEN = 30
RES = 301
N_EPOCHS = 5000
N_TRAIN = 90
MAKE_GIF = True
MAX_TOL = 0.5
MIN_TOL = 1e-10
XLIM = [-3,3]
YLIM = [-3,3]
PATIENCE = 50
DEFAULT_TOL = 1e-6
tol = DEFAULT_TOL
strikes = 0


### ~~~
## ~~~ Instantiate a ReLU network of depth L and constant width w
### ~~~

L = 2
w = 7
list_of_layers = [    
        nn.Linear(2, w),
        nn.ReLU(),
    ] + (L-1)*[
        nn.Linear(w, w),
        nn.ReLU(),
    ] + [
        nn.Linear(w, 1),
    ]
model = nn.Sequential(*list_of_layers)
cell_viz( model, xlim=XLIM, ylim=YLIM )



### ~~~
## ~~~ Make up a fake dataset
### ~~~

def f(x):
    if isinstance(x, torch.Tensor):
        return torch.cos( torch.pi*(x**2).sum(axis=1,keepdims=True)/6 )
    else:
        return np.cos( np.pi*(x**2).sum(axis=1,keepdims=True)/6 )

cell_viz( f, xlim=XLIM, ylim=YLIM )
x_test  = cartesian_product( torch.linspace(XLIM[0],XLIM[1],RES), torch.linspace(YLIM[0],YLIM[1],RES) )
x_train = x_test[torch.randperm(x_test.shape[0])[:N_TRAIN]]
y_train = f(x_train)



### ~~~
## ~~~ Visulize the model after training
### ~~~

optimizer = torch.optim.Adam( model.parameters(), lr=0.001 )
loss_fn = nn.MSELoss()
methods = ( "highs", "highs-ipm", "highs-ds", "cvxpy" )
data_on_solvers = { method:0 for method in methods }
if MAKE_GIF:
    gif = GifMaker( f"w={w}, e={N_EPOCHS}", ram_only=False, live_frame_duration=None )

with support_for_progress_bars():
    for i in trange(N_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if (i+1)%HOW_OFTEN==0 and (COMPUTE_SEGMENTS or MAKE_GIF):
            try:
                failed_attempts = 0
                while True:
                    try:
                        plt.close("all")
                        method = methods[min( failed_attempts, len(methods)-1 )]
                        fig, ax = minimalist_heatmap_where_relu_net_is_not_smooth(
                                model,
                                x_test,
                                verbose = False,
                                show = False,
                                tol = tol,
                                color = "black",
                                method = method
                            )
                        _ = ax.scatter( x_train[:,0].numpy(), x_train[:,1].numpy(), c="green", s=10 )
                        strikes = 0
                        tol *= 0.2
                        if tol<MIN_TOL:
                            tol = MIN_TOL
                        break
                    except:
                        # if failed_attempts==0:
                        #     print("highs didn't work")
                        failed_attempts += 1
                        tol *= 1.1
                        if tol > MAX_TOL:
                            tol = MAX_TOL
                            raise
                _ = ax.set_xlim(XLIM)
                _ = ax.set_ylim(YLIM)
                fig.tight_layout()
                if MAKE_GIF:
                    gif.capture()
                    data_on_solvers[method] += 1
                else:
                    plt.show()
            except:
                strikes += 1
                tol = DEFAULT_TOL
                if strikes >= PATIENCE:
                    raise
        plt.close("all")

if MAKE_GIF:
    gif.develop( fps=30, cleanup=False )

cell_viz( model, xlim=XLIM, ylim=YLIM )
