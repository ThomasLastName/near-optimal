
import math
import numpy as np
import cvxpy as cvx
from tqdm import tqdm
from matplotlib import pyplot as plt
from quality_of_life.my_plt_utils import points_with_curves, GifMaker
from quality_of_life.my_base_utils import support_for_progress_bars

class RigorousNet():
    def __init__( self, x_train, y_train ):
        #
        # ~~~ Define the training data
        sorted_indices = np.argsort(x_train)
        self.x_train = x_train[sorted_indices]
        self.y_train = y_train[sorted_indices]
        m = len(x_train)
        k = int(m/2)
        self.k = k
        spread = x_train.max() - x_train.min()
        #
        # ~~~ Define parameters
        self.a = np.random.normal() / math.sqrt(spread)
        self.b = np.random.normal() / math.sqrt(spread)
        self.c = np.random.normal(size=(k-1,)) / math.sqrt(k-1)
        self.tau = np.random.normal(size=(k-1,))
        #
        # ~~~ Define constraints
        self.lower_bounds = x_train[ 2*(np.arange(k-1)+1)-1 ].squeeze()
        self.upper_bounds = x_train[ 2*(np.arange(k-1)+1)   ].squeeze()
        #
        # ~~~ Apply constraints
        self.project()
    #
    # ~~~ Define how to project onto the constraint set
    def project(self):
        self.tau = np.clip( self.tau, a_min=self.lower_bounds, a_max=self.upper_bounds )
    #
    # ~~~ Evaluate v(x)
    def __call__(self,x):
        x = np.array(x)
        original_shape = x.shape
        x = x.reshape(-1)
        largest_active_nodes = np.searchsorted( self.tau, x )   # ~~~ self.tau[largest_active_nodes[j]-1] <= x[j] < self.tau[largest_active_nodes[j]] for all j
        predictions = []
        for j in range(len(x)):
            prediction = self.a*x[j] + self.b + sum(
                    self.c[ell]*(x[j]-self.tau[ell]) for ell in range(largest_active_nodes[j])
                )
            predictions.append(prediction)
        predictions = np.array(predictions).reshape(original_shape)
        if predictions.shape == (): predictions = predictions.item()
        return predictions
    #
    # ~~~ Minimize the training loss as a function of only a,b,c with tau treated as fixed
    def improve_coefficients( self, *args, **kwargs ):
        #
        # ~~~ Define the problem
        k = self.k
        t = cvx.Variable()
        a = cvx.Variable()
        b = cvx.Variable()
        c = cvx.Variable(k-1)
        objective = cvx.Minimize(t)
        constraints = self.create_epigraph_constraints( t, a, b, c, self.tau )
        #
        # ~~~ Solve it
        problem = cvx.Problem(objective,constraints)
        problem.solve(*args,**kwargs)
        self.a = a.value
        self.b = b.value
        self.c = c.value
        return problem
    #
    # ~~~ Minimize the training loss as a function of only a,b,c with tau treated as fixed
    def improve_breakpoints( self, *args, **kwargs ):
        #
        # ~~~ Define the problem
        k = self.k
        t = cvx.Variable()
        tau = cvx.Variable(k-1)
        objective = cvx.Minimize(t)
        constraints = self.create_epigraph_constraints( t, self.a, self.b, self.c, tau )
        for ell in range(k-1):
            constraints.append( tau[ell] >= self.lower_bounds[ell] )
            constraints.append( tau[ell] <= self.upper_bounds[ell] )
        #
        # ~~~ Solve it
        problem = cvx.Problem(objective,constraints)
        problem.solve(*args,**kwargs)
        self.tau = tau.value
        return problem
    #
    # ~~~ Create the constraints for the epigraph formulation
    def create_epigraph_constraints( self, t, a, b, c, tau ):
        k = self.k
        constraints = []
        for j in range(k):
            for i in [1,0]:
                j += 1  # ~~~ use 1-based indexing j=1,...,k
                index_2j_minus_i = (2*j-i)-1
                j -= 1  # ~~~ return to 0-based indexing
                evaluation_site = self.x_train[index_2j_minus_i].item()  # ~~~ == x_{2j-i}
                training_label  = self.y_train[index_2j_minus_i].item()  # ~~~ == y_{2j-i}
                #
                # ~~~ Compute the model's prediction at this data site
                prediction = a*evaluation_site + b + sum( c[ell]*(evaluation_site-tau[ell]) for ell in range(j) )
                #
                # ~~~ Add the epigraph constraints for abs(prediction-training_label) <= t
                constraints.append( prediction - training_label <= t )
                constraints.append( training_label - prediction <= t )
        return constraints


if __name__=="__main__":
    #
    # ~~~ Config
    from near_optimal.quadratic_univar import f, x_train, y_train
    x_train = x_train.numpy().squeeze()
    y_train = y_train.numpy().squeeze()
    v = RigorousNet( x_train, y_train )
    #
    # ~~~ Train
    N = 100
    how_often = 1
    history = []
    fig, ax = points_with_curves( x=x_train,  y=y_train, curves=(v,f), title=r"$\ell^\infty$ Error Minimization with Hard Constraints", show=False )
    gif = GifMaker()
    gif.capture()
    with support_for_progress_bars():
        pbar = tqdm( desc="Using Gradient Descent", total=N, ascii=' >=' )
        for i in range(N):
            max_error = v.improve_coefficients( solver=cvx.SCIPY, scipy_options={"method": "highs"} ) if i%2==0 else v.improve_breakpoints( solver=cvx.SCIPY, scipy_options={"method": "highs"} )
            max_error = max_error.objective.value
            _ = pbar.update()
            history.append(max_error)
            pbar.set_postfix({ "max_error" : f"{history[-1]:<4.4f}" })
            if (i+1)%how_often==0:
                fig, ax = points_with_curves( x=x_train,  y=y_train, curves=(v,f), title=r"$\ell^\infty$ Error Minimization with Hard Constraints", show=False, fig=fig, ax=ax )
                gif.capture()
    pbar.close()
    points_with_curves( x=x_train, y=y_train, curves=(v,f), title=r"$\ell^\infty$ Error Minimization with Hard Constraints", fig=fig, ax=ax )
    gif.develop()
