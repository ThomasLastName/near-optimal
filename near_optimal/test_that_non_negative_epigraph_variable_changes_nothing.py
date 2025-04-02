
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from itertools import product
from quality_of_life.my_base_utils import support_for_progress_bars

from near_optimal.quadratic_univar import x_train, y_train, k, m, DualSpline
v = DualSpline( x_train, y_train )

WEIGHTED_MEAN = ( True, False )
T_SQUARED_CONSTRAINT = ( True, False )
T_SQUARED_OBJECTIVE = ( True, False )
MSE_PENALTY = np.linspace( 0, 2*m, 31 )
TOL = ( 1e-6, 1e-8 )
BREAKPOINT_REG = ( 0, 1e-12, 1e-8, 1e-4 )


differences = []
with support_for_progress_bars():
    for ( mse_penalty, tol, breakpoint_reg, t_squared_objective, t_squared_constraint, weighted_mean ) in tqdm( product(MSE_PENALTY,TOL,BREAKPOINT_REG,T_SQUARED_OBJECTIVE,T_SQUARED_CONSTRAINT,WEIGHTED_MEAN), total=len(MSE_PENALTY)*len(TOL)*len(BREAKPOINT_REG)*len(T_SQUARED_OBJECTIVE)*len(T_SQUARED_CONSTRAINT)*len(WEIGHTED_MEAN) ):
        if not (t_squared_constraint and not t_squared_objective):
            if not (weighted_mean and mse_penalty==0):
                false = v.S_Lemma_1( mse_penalty=mse_penalty, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=t_squared_constraint, weighted_mean=weighted_mean, print_info=False, non_negative_epigraph=False )
                true  = v.S_Lemma_1( mse_penalty=mse_penalty, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=t_squared_constraint, weighted_mean=weighted_mean, print_info=False, non_negative_epigraph=True  )
                differences.append(abs(false-true))
                false = v.S_Lemma_2( mse_penalty=mse_penalty, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=t_squared_constraint, weighted_mean=weighted_mean, print_info=False, non_negative_epigraph=False )
                true  = v.S_Lemma_2( mse_penalty=mse_penalty, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=t_squared_constraint, weighted_mean=weighted_mean, print_info=False, non_negative_epigraph=True  )
                differences.append(abs(false-true))
