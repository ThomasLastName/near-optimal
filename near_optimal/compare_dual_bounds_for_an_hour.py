"""
Since there are so many different ways to formulate the problem as a quadratic program,
in this file, we just test a bunch of them to get a sense of what works and what doesn't.
Note, the figure is *NOT* presentation-ready. It's just for my own needs.
"""

import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import pandas as pd
from itertools import product
from time import time
from quality_of_life.my_base_utils import support_for_progress_bars


from near_optimal.quadratic_univar import x_train, y_train, DualSpline

#
# ~~~ Settings
MSE_PENALTY = np.linspace( 0, 8, 25 )
TOL = ( 1e-6, 1e-9 )
BREAKPOINT_REG = ( 0, 1e-12, 1e-8 )
T_SQUARED_OBJECTIVE = ( True, False )
Z_SQUARED_OBJECTIVE = ( True, False )

#
# ~~~ Helper function
def are_equal(a,b):
    try: return abs(a-b)<1e-13
    except: return a==b

#
# ~~~ Load data
try:    data = pd.read_csv("comparison_data.csv").to_dict(orient="records")
except: data = []

start_time = time()
v = DualSpline( x_train, y_train )
with support_for_progress_bars():
    for mse_penalty, tol, breakpoint_reg, t_squared_objective, z_squared_objective in tqdm(
                product( MSE_PENALTY, TOL, BREAKPOINT_REG, T_SQUARED_OBJECTIVE, Z_SQUARED_OBJECTIVE ),
                total = len(MSE_PENALTY)*len(TOL)*len(BREAKPOINT_REG)*len(T_SQUARED_OBJECTIVE)*len(Z_SQUARED_OBJECTIVE),
                desc = "Computing a Whole Bunch of Lower Bounds",
                ascii = " >="
        ):
        if time() - start_time > 3600:
            print("An hour is up!")
            break
        else:
            settings = { "mse_penalty":mse_penalty, "tol":tol, "breakpoint_reg":breakpoint_reg, "t_squared_objective":t_squared_objective, "z_squared_objective":z_squared_objective }
            already_computed_it = any(
                    all( are_equal(d[key],value) for key, value in settings.items()) 
                    for d in data
                )
            if not already_computed_it:
                lower_bound = v.solve_dual_of_mse_minimization_with_more_options( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, z_squared_constraint=z_squared_objective, weighted_mean=(mse_penalty>0), print_info=False )
                # _           = v.solve_dual_of_mse_minimization_with_more_options( mse_penalty=mse_penalty, solver=cvx.SCS, eps_abs=tol, eps_rel=tol, eps_infeas=tol/1000, breakpoint_reg=breakpoint_reg, t_squared_objective=t_squared_objective, t_squared_constraint=False, z_squared_constraint=z_squared_objective, weighted_mean=False,           print_info=False )
                upper_bound = (v.z-v.y).abs().max().item() if v.compute_violation().min().item() > 1 - 1e-8 else None
                data.append({
                        "mse_penalty" : mse_penalty,
                        "tol" : tol,
                        "breakpoint_reg" : breakpoint_reg,
                        "t_squared_objective" : t_squared_objective,
                        "z_squared_objective" : z_squared_objective,
                        "lower_bound" : lower_bound,
                        "upper_bound" : upper_bound
                    })

data = pd.DataFrame(data)
data.to_csv("comparison_data.csv", index=False )

#
# ~~~ Convert bools to string for easier grouping
data["t_squared_objective"] = data["t_squared_objective"].astype(str)
data["z_squared_objective"] = data["z_squared_objective"].astype(str)

#
# ~~~ Define colors for each unique (breakpoint_reg, t_squared_objective) combination
unique_combinations = data[["breakpoint_reg", "t_squared_objective", "z_squared_objective"]].drop_duplicates()
palette = sns.color_palette("tab10", len(unique_combinations))
color_map = {tuple(row): palette[i] for i, row in enumerate(unique_combinations.itertuples(index=False, name=None))}

#
# ~~~ Define line styles for different `tol` values
linestyle_map = {1e-9: "-", 1e-6: "--"}
_ = plt.figure(figsize=(10, 6))

#
# ~~~ Plot each group separately
for (breakpoint_reg, t_squared_objective, tol, z_squared_objective), group in data.groupby(["breakpoint_reg", "t_squared_objective", "tol", "z_squared_objective"]):
    color = color_map[(breakpoint_reg, t_squared_objective, z_squared_objective)]
    linestyle = linestyle_map[group["tol"].iloc[0]]
    #
    # ~~~ Plot lower bound
    _ = plt.plot(group["mse_penalty"], group["lower_bound"], color=color, linestyle=linestyle, label=f"({breakpoint_reg}, {t_squared_objective}, {z_squared_objective}, tol={group['tol'].iloc[0]})")
    #
    # ~~~ Plot upper bound using the same color
    _ = plt.plot(group["mse_penalty"], group["upper_bound"], color=color, linestyle=linestyle)

_ = plt.xlabel("MSE Penalty")
_ = plt.ylabel("Bound Values")
_ = plt.title(f"Lower and Upper Bounds vs. MSE Penalty")
_ = plt.legend( title="(breakpoint_reg, t_squared_objective, tol)", loc="lower right" )
_ = plt.ylim(0, 0.1)  # Set y-axis limits
_ = plt.grid(True)
_ = plt.show()

