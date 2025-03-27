
import torch
import cvxpy as cvx
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from itertools import product
from quality_of_life.ansi import bcolors
from quality_of_life.my_base_utils import support_for_progress_bars


from near_optimal.quadratic_univar import x_train, y_train, k, m, DualSpline
v = DualSpline( x_train, y_train )

#
# ~~~ Compute the first dual lower bound that I came up with
mse_lower_bound = np.sqrt(v.S_Lemma_3(print_info=False).objective.value)


# NOTE: S_Lemma_( ..., t_squared_constraint=B, t_squared_objective=B, ... ) appears to give identical results for B=True and B=False, regardless of other kwargs (the ...)

#
# ~~~ 
EPIGRAPH_OBJECTIVE = ("linear", "quadratic")
EPIGRAPH_CONSTRAINT = ("linear", "quadratic")

MSE_PENALTY = np.linspace( 0, 2*m, 101 )

TOL = ( 1e-6, 1e-9 )
BREAKPOINT_REG = ( 0, 1e-12, 1e-8, 1e-4 )

data = []

with support_for_progress_bars():
    for mse_penalty in MSE_PENALTY:
        print("")
        print(f"    Testing mse_penalty={bcolors.HEADER + str(mse_penalty) + bcolors.OKGREEN} (will test up to {MSE_PENALTY.max()})")
        print("")
        lst = []
        for tol, breakpoint_reg in tqdm( product(TOL,BREAKPOINT_REG), total=len(TOL)*len(BREAKPOINT_REG), desc="MSE Minimization", ascii=" >=" ):
            #
            # ~~~ Solve the dual
            dual_max = v.S_Lemma_3( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, print_info=False ).objective.value
            #
            # ~~~ Append the results as a new row in the DataFrame
            lst.append(dual_max)
        data.append(lst)
            # data.append({
            #         "mse_penalty": mse_penalty,
            #         "tol": tol,
            #         "breakpoint_reg": breakpoint_reg,
            #         "dual_max": dual_max
            #     })

data = np.array(data)
for j in range(data.shape[1]):
    plt.plot( MSE_PENALTY[:data.shape[0]], data[:,j])

plt.grid()
plt.show()
