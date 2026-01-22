"""
This is basically the same as https://colab.research.google.com/drive/1C6Xgo9C-U-ZTcxDtap7I44Ao-Pd785Fq?usp=sharing
"""

import torch
from torch import nn
from tqdm.auto import tqdm
from quality_of_life.my_plt_utils import points_with_curves, GifMaker
from quality_of_life.my_base_utils import support_for_progress_bars

class RigorousNet(nn.Module):
    def __init__(self,x_train):
        super().__init__()
        self.x_train = x_train.sort().values
        m = len(x_train)
        k = int(m/2)
        self.relu_net = nn.Sequential(
            nn.Linear(1,k-1),
            nn.ReLU(),
            nn.Linear(k-1,1)
        )
        self.a = nn.Parameter( torch.randn(1) )
        self.b = nn.Parameter( torch.randn(1) )
        self.relu_net[0].weight.requires_grad = False
        self.relu_net[0].weight.fill_(1.)
        self.lower_bounds = x_train[ 2*(torch.arange(k-1)+1)-1 ].squeeze()
        self.upper_bounds = x_train[ 2*(torch.arange(k-1)+1)   ].squeeze()
        self.project()
    def project(self):
        with torch.no_grad():
            self.relu_net[0].bias.data.clamp_( min=-self.upper_bounds, max=-self.lower_bounds )
    def forward(self,x):
        return self.relu_net(x) + self.a*x + self.b


if __name__=="__main__":
    #
    # ~~~ Config
    torch.manual_seed(2024)
    # k = 5
    # m =  2*k
    # f = lambda x: torch.sin(2*torch.pi*x)
    # scale = 0.1
    # x_train = torch.linspace(-1,1,m).reshape(-1,1)
    # y_train = f(x_train) + scale*torch.randn_like(x_train)
    # x_test = torch.linspace(-1,1,1001)
    # y_test = f(x_test)
    from near_optimal.quadratic_univar import f, x_train, y_train
    x_train = x_train.reshape(-1,1)
    y_train = y_train.reshape(-1,1)
    #
    # ~~~ Create and train the model
    model = RigorousNet(x_train)
    lr = 1e-2
    N = 20000
    how_often = 100
    optimizer = torch.optim.Adam( model.parameters(), lr=lr )
    scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size=5000, gamma=0.3 )
    #
    # ~~~ Gradient Descent
    history = []
    fig, ax = points_with_curves( x=x_train.squeeze(),  y=y_train.squeeze(), grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(model,f), title=r"$\ell^\infty$ Error Minimization with Hard Constraints (Projected Gradient Descent)", show=False )
    gif = GifMaker()
    gif.capture()
    best_error = torch.inf
    with support_for_progress_bars():
        pbar = tqdm( desc="Using Gradient Descent", total=N, ascii=' >=' )
        for i in range(N):
            predictinons = model(x_train)
            max_error = (y_train-predictinons).abs().max()
            max_error.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.project()
            best_error = min( best_error, max_error.item() )
            _ = pbar.update()
            history.append(max_error.item())
            pbar.set_postfix({
                    "max_error" : f"{history[-1]:<4.4f}",
                    "best_error" : f"{best_error:<4.4f}"
                })
            if (i+1)%how_often==0:
                fig, ax = points_with_curves( x=x_train.squeeze(),  y=y_train.squeeze(), grid=torch.linspace(-1,1,501).reshape(-1,1), curves=(model,f), title=r"$\ell^\infty$ Error Minimization with Hard Constraints", show=False, fig=fig, ax=ax )
                gif.capture()
    pbar.close()
    gif.develop()
