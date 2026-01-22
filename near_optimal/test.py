
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import pandas as pd
import random
from time import time
from quality_of_life.my_base_utils import support_for_progress_bars


from near_optimal.quadratic_univar import x_train, y_train, m, DualSpline
v = DualSpline( x_train, y_train )

# weighted_mse_lower_bound = v.solve_dual_of_mse_minimization( weighted_mean=True, print_info=False )

#
# ~~~ 
MSE_PENALTY = np.linspace( 0, 8, 25 )
TOL = ( 1e-6, 1e-9 )
BREAKPOINT_REG = ( 0, 1e-12, 1e-8 )
T_SQUARED_OBJECTIVE = ( True, False )

def are_equal(a,b):
    try: return abs(a-b)<1e-13
    except: return a==b

differences = []
start_time = time()
while time() - start_time < 3600:
    mse_penalty = random.choice(MSE_PENALTY)
    tol = random.choice(TOL)
    breakpoint_reg = random.choice(BREAKPOINT_REG)
    t_squared_objective = random.choice(T_SQUARED_OBJECTIVE)
    old = v.S_Lemma_1( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=False, print_info=False )
    new = v.solve_dual_of_mse_minimization_with_more_options( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=False, print_info=False, z_squared_constraint=True )
    differences.append(abs(new-old))
    old = v.S_Lemma_1( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=(mse_penalty>0), print_info=False )
    new = v.solve_dual_of_mse_minimization_with_more_options( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=(mse_penalty>0), print_info=False, z_squared_constraint=True )
    differences.append(abs(new-old))
    old = v.S_Lemma_2( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=False, print_info=False )
    new = v.solve_dual_of_mse_minimization_with_more_options( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=False, print_info=False, z_squared_constraint=False )
    differences.append(abs(new-old))
    old = v.S_Lemma_2( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=(mse_penalty>0), print_info=False )
    new = v.solve_dual_of_mse_minimization_with_more_options( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=(mse_penalty>0), print_info=False, z_squared_constraint=False )
    differences.append(abs(new-old))
    print("")
    print(differences)
    print("")
