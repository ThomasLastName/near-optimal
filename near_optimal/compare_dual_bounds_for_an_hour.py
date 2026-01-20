
import torch
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import pandas as pd
from itertools import product
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

try:
    data_1 = pd.read_csv("comparison_data_1.csv").to_dict(orient="records")
except FileNotFoundError:
    data_1 = []

try:
    data_2 = pd.read_csv("comparison_data_2.csv").to_dict(orient="records")
except FileNotFoundError:
    data_2 = []

start_time = time()
with support_for_progress_bars():
    for mse_penalty, tol, breakpoint_reg, t_squared_objective in tqdm(
                product( MSE_PENALTY, TOL, BREAKPOINT_REG, T_SQUARED_OBJECTIVE ),
                total = len(MSE_PENALTY)*len(TOL)*len(BREAKPOINT_REG)*len(T_SQUARED_OBJECTIVE),
                desc = "Computing a Whole Bunch of Lower Bounds",
                ascii = " >="
        ):
        if time() - start_time > 3600:
            print("An hour is up!")
            break
        else:
            settings = { "mse_penalty":mse_penalty, "tol":tol, "breakpoint_reg":breakpoint_reg, "t_squared_objective":t_squared_objective }
            already_computed_it = any(
                    all( are_equal(d[key],value) for key, value in settings.items()) 
                    for d in data_1 + data_2  # Assuming you want to check in both lists
                )
            if not already_computed_it:
                # print(settings)
                for data, function in zip( [data_1,data_2], [v.S_Lemma_1,v.S_Lemma_2] ):
                    lower_bound = function( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=(mse_penalty>0), print_info=False )
                    _           = function( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, weighted_mean=False,           print_info=False )
                    upper_bound = (v.z-v.y).abs().max().item() if v.compute_violation().min().item() > 1 - 1e-8 else None
                    data.append({
                            "mse_penalty" : mse_penalty,
                            "tol" : tol,
                            "breakpoint_reg" : breakpoint_reg,
                            "t_squared_objective" : t_squared_objective,
                            "lower_bound" : lower_bound,
                            "upper_bound" : upper_bound
                        })

data_1 = pd.DataFrame(data_1)
data_2 = pd.DataFrame(data_2)
data_1.to_csv("comparison_data_1.csv", index=False )
data_2.to_csv("comparison_data_2.csv", index=False )


method = "S_Lemma_1"
for data in (data_1,data_2):
    #
    # ~~~ Convert `t_squared_objective` to a string for easier grouping
    data["t_squared_objective"] = data["t_squared_objective"].astype(str)
    #
    # ~~~ Define colors for each unique (breakpoint_reg, t_squared_objective) combination
    unique_combinations = data[["breakpoint_reg", "t_squared_objective"]].drop_duplicates()
    palette = sns.color_palette("tab10", len(unique_combinations))
    color_map = {tuple(row): palette[i] for i, row in enumerate(unique_combinations.itertuples(index=False, name=None))}
    #
    # ~~~ Define line styles for different `tol` values
    linestyle_map = {1e-9: "-", 1e-6: "--"}
    plt.figure(figsize=(10, 6))
    #
    # ~~~ Plot each group separately
    for (breakpoint_reg, t_squared_objective, tol), group in data.groupby(["breakpoint_reg", "t_squared_objective", "tol"]):
        color = color_map[(breakpoint_reg, t_squared_objective)]
        linestyle = linestyle_map[group["tol"].iloc[0]]
        #
        # ~~~ Plot lower bound
        plt.plot(group["mse_penalty"], group["lower_bound"], color=color, linestyle=linestyle, label=f"({breakpoint_reg}, {t_squared_objective}, tol={group['tol'].iloc[0]})")
        #
        # ~~~ Plot upper bound using the same color
        plt.plot(group["mse_penalty"], group["upper_bound"], color=color, linestyle=linestyle)
    plt.xlabel("MSE Penalty")
    plt.ylabel("Bound Values")
    plt.title(f"Lower and Upper Bounds vs. MSE Penalty ({method})")
    plt.legend(title="(breakpoint_reg, t_squared_objective, tol)")
    plt.ylim(0, 0.1)  # Set y-axis limits
    plt.grid(True)
    plt.show()
    method = "S_Lemma_2"

