
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
mse_lower_bound = np.sqrt(v.solve_dual_of_mse_minimization(print_info=False).objective.value)
# v.S_Lemma_2( mse_penalty=2, t_squared_constraint=False, t_squared_objective=True, eps_abs=1e-6, eps_rel=1e-6 ).objective.value

# NOTE: S_Lemma_( ..., t_squared_constraint=B, t_squared_objective=B, ... ) appears to give identical results for B=True and B=False, regardless of other kwargs (the ...)

#
# ~~~ 
MSE_PENALTY = np.linspace( 0, 2*m, 51 )
TOL = ( 1e-6, 1e-9 )
BREAKPOINT_REG = ( 0, 1e-12, 1e-8, 1e-4 )
T_SQUARED_CONSTRAINT = ( True, False )

data_1 = []
data_2 = []

with support_for_progress_bars():
    for mse_penalty, tol, breakpoint_reg, t_squared_constraint in tqdm(
                product( MSE_PENALTY, TOL, BREAKPOINT_REG, T_SQUARED_CONSTRAINT ),
                total = len(MSE_PENALTY)*len(TOL)*len(BREAKPOINT_REG)*len(T_SQUARED_CONSTRAINT),
                desc = "Computing a Whole Bunch of Lower Bounds",
                ascii = " >="
            ):
            for data, function in zip( [data_1,data_2], [v.S_Lemma_1,v.S_Lemma_2] ):
                lower_bound = function( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_constraint=t_squared_constraint, t_squared_objective=True, print_info=False ).objective.value
                data.append({
                        "mse_penalty" : mse_penalty,
                        "tol" : tol,
                        "breakpoint_reg" : breakpoint_reg,
                        "lower_bound" : lower_bound
                    })

data = np.array(data)
for j in range(data.shape[1]):
    plt.plot( MSE_PENALTY[:data.shape[0]], data[:,j])

plt.grid()
plt.show()
