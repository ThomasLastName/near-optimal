
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from itertools import product
from quality_of_life.my_base_utils import support_for_progress_bars


from near_optimal.quadratic_univar import x_train, y_train, k, m, DualSpline
v = DualSpline( x_train, y_train )


NON_NEGATIVE_EPIGRAPH = ( True, False )
MSE_PENALTY = np.linspace( 0, 2*m, 31 )
TOL = ( 1e-6, 1e-8 )
BREAKPOINT_REG = ( 0, 1e-12, 1e-8, 1e-4 )

differences = []
with support_for_progress_bars():
    for ( mse_penalty, tol, breakpoint_reg, non_negative_epigraph ) in tqdm( product(MSE_PENALTY,TOL,BREAKPOINT_REG,NON_NEGATIVE_EPIGRAPH), total=len(MSE_PENALTY)*len(TOL)*len(BREAKPOINT_REG)*len(NON_NEGATIVE_EPIGRAPH) ):
        false_false = v.S_Lemma_1( t_squared_objective=False, t_squared_constraint=False, mse_penalty=mse_penalty, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, print_info=False ).objective.value
        true_true   = v.S_Lemma_1( t_squared_objective=True,  t_squared_constraint=True,  mse_penalty=mse_penalty, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, print_info=False ).objective.value
        differences.append(abs(false_false-true_true))
        false_false = v.S_Lemma_2( t_squared_objective=False, t_squared_constraint=False, mse_penalty=mse_penalty, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, print_info=False ).objective.value
        true_true   = v.S_Lemma_2( t_squared_objective=True,  t_squared_constraint=True,  mse_penalty=mse_penalty, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, print_info=False ).objective.value
        differences.append(abs(false_false-true_true))

# NOTE: if max(TOL)=1e-6, then max(differences) is like 1e-5, so bsaically we conclude that S_Lemma_( ..., False, False, ... ) and S_Lemma_( ..., True, True, ... ) give identical results
# NOTE: empirically, min(differences) is always positive, suggesting that in general S_Lemma_( ..., False, False, ... ) > S_Lemma_( ..., True, True, ... ), implying that latter is more conservative, and less likely to return a value erroneously exceeding the primal minimum
