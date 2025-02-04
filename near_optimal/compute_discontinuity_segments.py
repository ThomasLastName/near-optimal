
import math
import numpy as np
import cvxpy as cp
import torch
from tqdm import trange, tqdm
from math import comb as binom
from scipy.optimize import linprog
from matplotlib import pyplot as plt
from quality_of_life.my_torch_utils import cartesian_product
from quality_of_life.my_plotly_utils import cell_viz
from quality_of_life.my_base_utils import buffer
from plotly import graph_objects as go

try:    # ~~~ these functions help control the color of console output; find at https://github.com/ThomasLastName/quality_of_life
    from quality_of_life.my_base_utils import support_for_progress_bars
except: # ~~~ however, if those functions are not available, then let their definitions be trivial (for compatibility)
    from contextlib import contextmanager
    @contextmanager
    def support_for_progress_bars():
        yield

#
# ~~~ Minimize abs(a_new@x_new + b_new) subject to `eps[i]*(A[i]@x+b[i])>=0` for all i
def minimize_abs_linear_function_on_polygon( a_new, b_new, A, b, eps, return_x=False, tol=None, method="highs" ):
    k,d = A.shape
    try:
        if np.allclose(a_new,0):
            assert b_new==0
        if method=="cvxpy":
            x = cp.Variable(d)
            inequality_constraints = [
                    eps[i] * (A[i]@x + b[i]) >= 0
                    for i in range(k)
                ]
            problem = cp.Problem( cp.Minimize(0), inequality_constraints+[ 0 == a_new@x + b_new ] )
            if tol is None:
                problem.solve(solver=cp.ECOS)
            else:
                problem.solve( solver=cp.ECOS, abstol=tol, reltol=tol, feastol=tol )
            yes_it_is_feasible = not (problem.status == cp.INFEASIBLE)
            assert yes_it_is_feasible
            assert np.sqrt(np.mean(np.abs(x.value**2))) < 200
            return ( 0., x.value ) if return_x else 0.
        else:
            solved_problem = linprog(
                    c = np.zeros(d),
                    #
                    # ~~~ -Ax \leq b \iff Ax+b \geq 0
                    A_ub = np.row_stack([ eps[i]*(-A[i]) for i in range(k) ]),
                    b_ub = np.row_stack([ eps[i] * b[i] for i in range(k) ]),
                    A_eq = np.row_stack([ a_new ]),
                    b_eq = np.row_stack([-b_new ]),
                    bounds = (None,None),
                    method = method
                )
            yes_it_is_feasible = not (solved_problem.status == 2)
            assert yes_it_is_feasible
            assert np.sqrt(np.mean(np.abs(solved_problem.x**2))) < 200
            return ( 0., solved_problem.x ) if return_x else 0.
    except AssertionError:
        if method=="cvxpy":
            x = cp.Variable(d)
            objective = cp.Minimize(cp.abs(a_new @ x + b_new))
            inequality_constraints = [
                    eps[i] * (A[i]@x + b[i]) >= 0
                    for i in range(k)
                ]
            problem = cp.Problem( objective, inequality_constraints )
            if tol is None:
                problem.solve(solver=cp.ECOS)
            else:
                problem.solve( solver=cp.ECOS, abstol=tol, reltol=tol, feastol=tol )
            return ( a_new @ x.value + b_new, x.value ) if return_x else a_new @ x.value + b_new
        else:
            solved_problem = linprog(
                    c = np.array(d*[0.]+[1.]),  # ~~~ c == (0,...,0,1) length d+1; minimize the epigraph variable
                    #
                    # ~~~ -Ax \leq b \iff Ax+b \geq 0
                    A_ub = np.row_stack(
                            [
                                np.concatenate([ eps[i]*(-A[i]), [0.] ])
                                for i in range(k)
                            ] +
                            [ np.concatenate([ a_new, [-1.] ]) ] + 
                            [ np.concatenate([-a_new, [-1.] ]) ]
                        ),
                    b_ub = np.row_stack(
                            [ eps[i] * b[i] for i in range(k) ] + 
                            [-b_new] +
                            [ b_new]
                        ),
                    bounds = (None,None),
                    method = method
                )
            # print(solved_problem)
            minimized_abs_linear_function_on_polygon = solved_problem.x[-1]
            x = solved_problem.x[:d]
            assert abs( abs(a_new@x+b_new) - minimized_abs_linear_function_on_polygon ) < 1e-6
            return ( a_new@x+b_new, x ) if return_x else a_new@x+b_new

#
# ~~~ Derive the sign patterns that are necessary to describe the j=1,...,1+binom(n+1,2) regions between n lines `A[i].T@x+b[i]==0` (i=1,...,n) via `\eps[i,j]*(A[i].T@x+b[i])>=0`
def derive_signs_for_linear_constraints_of_a_shallow_net( A, b, verbose=True, tol=None, desc=None, method="highs" ):
    n,d = A.shape
    assert b.shape==(n,)
    assert n>=2
    if not d>1:
        raise NotImplementedError("This function is only implemented for d>1")
    relevant_sign_patterns = [
            [  1,  1 ],
            [  1,- 1 ],
            [ -1,  1 ],
            [ -1,- 1 ]
        ]
    with support_for_progress_bars():
        iterator = trange( 2, n, ascii=" >=", initial=2, total=n, desc=("Deriving Linear Pieces" if desc is None else desc) ) if verbose else range(2,n)
        for k in iterator:
            # print(k)
            new_sign_patterns = []
            a_new, b_new = A[k], b[k]
            # print(relevant_sign_patterns)
            for eps in relevant_sign_patterns:
                #
                # ~~~ Test whether or not the line `a_new@x+b_new==0` intersects the polygon defined by `eps[i]*(A[i]@x+b[i])>=0` for all i<k 
                signed_minimal_abs = minimize_abs_linear_function_on_polygon( a_new, b_new, A[:k], b[:k], eps, tol=tol )
                # print(signed_minimal_abs)
                #
                # ~~~ If a_new @ x + b_new==0 at the solution, it means polygon (determined by eps) is bifurcated by the new line, it splits into two polygons
                if signed_minimal_abs == 0:
                    new_sign_patterns.append( eps + [1] )
                    new_sign_patterns.append( eps + [-1])
                else:
                    new_sign_patterns.append( eps + [1 if (signed_minimal_abs>0) else -1] )
            #
            # ~~~ Wrap up
            relevant_sign_patterns = new_sign_patterns
            example_of_a_sign_pattern = relevant_sign_patterns[0]
            number_of_lines_or_hyperplanes = len(example_of_a_sign_pattern)
            assert number_of_lines_or_hyperplanes == k+1
            # print(len(relevant_sign_patterns))
            # print(sum( binom(number_of_lines_or_hyperplanes,j) for j in range(d+1) ))
            assert len(relevant_sign_patterns) == sum( binom(number_of_lines_or_hyperplanes,j) for j in range(d+1) )  # ~~~ the maximal number of regions in R^d formed by n hyperplanes is apprently sum_{j=0}^d binom(n,j), which is also the maximal number of linear pieces of a shallow ReLU net of width n on R^d; this quantity is called "Lazy caterer's sequence" when d=2, and called the "cake sequence" when d=3
    return relevant_sign_patterns

#
# ~~~ Solve for the locus of points x where 0 == f(x) = d + \sum_{i=1}^n c_j*ReLU(A[i]@x+b[i])
def solve_for_where_shallow_relu_net_is_zero( A, b, c, d, sign_patterns, verbose=True, really_big_number=2026, tol=1e-5, regularize=True, method="highs" ):
    w, dim = A.shape
    c = c.squeeze()
    assert len(sign_patterns) == sum( binom(w,j) for j in range(dim+1) )  # ~~~ safety feature; check that the number of sign patterns (i.e., regions of linearity) is correct
    assert b.shape == (w,)
    try:
        assert c.shape == (w,)
    except:
        c = c.squeeze()
        assert c.shape == (w,)
    try:
        assert isinstance(d,float)
    except:
        d = d.item()
        assert isinstance(d,float)
    segments = []
    #
    # ~~~ Solve for the zero set of the neural network on this region
    with support_for_progress_bars():
        iterator = tqdm( sign_patterns, ascii=" >=", desc="Solving for Zero on Each Piece" ) if verbose else sign_patterns
        for eps in iterator:
            #
            # ~~~ On the region eps[i]*(a[i]@x + b[i]) > 0, we have that f(x) is affine; specifically, f(x) == a_new@x + b_new
            a_new = sum( c[i]*A[i] if eps[i]>0 else 0*A[i] for i in range(w) )
            b_new = sum( c[i]*b[i] if eps[i]>0 else 0*b[i] for i in range(w) ) + d
            #
            # ~~~ Check whether or not f(x) == a_new@x + b_new is ever zero on this region by minimizing abs(a_new@x + b_new)
            minimal_value, minimizer = minimize_abs_linear_function_on_polygon( a_new, b_new, A, b, eps, return_x=True )
            the_net_is_zero_at_some_point_in_this_region = ( minimal_value == 0 )
            if the_net_is_zero_at_some_point_in_this_region:# and np.linalg.norm(minimizer) < 1000:
                # print(eps)
                #
                # ~~~ Find two points x at which the line of a_new@x + b_new == 0 intersects the *boundary* of polygon defined by eps[i]*(A[i]@x+b[i])>=0
                for j in range(w):
                    #
                    # ~~~ Search for x on the piece A[j]@x+b[j]==0 of the boundary
                    if method == "cvxpy":
                        x = cp.Variable(dim)
                        constraints = [
                                eps[i] * (A[i]@x + b[i]) >= 0
                                for i in range(w)
                            ] + [
                                a_new@x + b_new == 0,
                                A[j]@x + b[j] == 0
                            ]
                        objective =  cp.Minimize(cp.norm(x)**2) if regularize else cp.Minimize(0)
                        problem = cp.Problem( objective, constraints ) # ~~~ we really just care about feasibility, but we minimize the norm squared for numerical stability
                        if tol is None:
                            problem.solve(solver=cp.ECOS)
                        else:
                            problem.solve( solver=cp.ECOS, abstol=tol, reltol=tol, feastol=tol )
                        # print(problem.status)
                        found_an_x_that_intersects_the_boundary = ( problem.status==cp.OPTIMAL or problem.status==cp.OPTIMAL_INACCURATE )
                        if found_an_x_that_intersects_the_boundary:
                            endpoint = x.value
                            break
                    else:
                        solved_problem = linprog(
                                c = np.zeros(dim),
                                #
                                # ~~~ -Ax \leq b \iff Ax+b \geq 0
                                A_ub = np.row_stack([ eps[i]*(-A[i]) for i in range(w) ]),
                                b_ub = np.row_stack([ eps[i] * b[i] for i in range(w) ]),
                                A_eq = np.row_stack([ a_new, A[j] ]),
                                b_eq = np.row_stack([-b_new,-b[j] ]),
                                bounds = (None,None),
                                method = method
                            )
                        found_an_x_that_intersects_the_boundary = (solved_problem.status==0)
                        if found_an_x_that_intersects_the_boundary:
                            endpoint = solved_problem.x
                            break
                #
                # ~~~ Now, `minimizer` and `end_point` form a line segment; let's make the line segment as long as possible within the region
                if method == "cvxpy":
                    t = cp.Variable(1,nonneg=True)  # ~~~ a non-negative scalar
                    x = cp.Variable(dim)            # ~~~ a point on the line segment beteen from `endpoint` to `minimizer`
                    constraints = [
                        x == endpoint + t*(minimizer-endpoint)
                    ] + [
                                eps[i] * (A[i]@x + b[i]) >= 0
                                for i in range(w)
                        ]
                    problem = cp.Problem( cp.Maximize(t), constraints )
                    if tol is None:
                        problem.solve(solver=cp.ECOS)
                    else:
                        problem.solve( solver=cp.ECOS, abstol=tol, reltol=tol, feastol=tol )
                    if problem.status == cp.UNBOUNDED:
                        other_endpoint = endpoint + really_big_number*(minimizer-endpoint) # ~~~ just some other point on the line segment outside the xlim and ylim we'll use
                    else:
                        other_endpoint = x.value
                else:
                    signed_A = np.row_stack([ eps[i]*A[i] for i in range(w) ])
                    signed_b = np.row_stack([ eps[i]*b[i] for i in range(w) ])
                    # print((-A@(minimizer-endpoint))[:,None])
                    # print((A@endpoint)[:,None] + b)
                    solved_problem = linprog(
                            c = -np.ones(1),
                            #
                            # ~~~ -Ax \leq b \iff Ax+b \geq 0
                            A_ub = (-signed_A@(minimizer-endpoint))[:,None],
                            b_ub = (signed_A@endpoint)[:,None] + signed_b,
                            bounds = (0,None),
                            method = method
                        )
                    if solved_problem.status==3:
                        other_endpoint = endpoint + really_big_number*(minimizer-endpoint)
                    else:
                        other_endpoint = endpoint + solved_problem.x[0]*(minimizer-endpoint)
                segments.append(( endpoint, other_endpoint ))
    return segments
            # #
            # # ~~~ Use ell^1 norm minimization as a heuristic to try to find a point at which the line of a_new@x + b_new == 0 intersects the boundary polygon defined by eps[i]*(A[i]@x+b[i])>=0, i.e., solves A[j]@x+b[j]==0 for some j
            # x = cp.Variable(dim)
            # y = cp.Variable(w)
            # constraints = [
            #         y[i] == eps[i] * (A[i]@x + b[i])
            #         for i in range(w)
            #     ] + [ y >= 0 ] + [ 0 == a_new@x + b_new ]
            # problem = cp.Problem( cp.Minimize(sum(y)), constraints )
            # problem.solve(solver=cp.ECOS)
            # print(y.value)
            # j = y.value.argmin()
            # print(j)
            # # one_endpoint = np.linalg.solve(A[j],-b[j])
            # #
            # # ~~~ Verify correctness
            # x = cp.Variable(dim)
            # constraints = [
            #         eps[i] * (A[i]@x + b[i]) >= 0
            #         for i in range(w)
            #     ] + [ A[j]@x + b[j] ==0 ] + [ 0 == a_new@x + b_new ]
            # problem = cp.Problem( cp.Minimize(cp.norm(x)**2), constraints )
            # problem.solve(solver=cp.ECOS)
            # correct = not (problem.status == cp.INFEASIBLE)
            # assert correct
            # one_endpoint = x.value
            # #
            # # ~~~
            # x = cp.Variable(dim)
            # y = cp.Variable(w)
            # constraints = [
            #         y[i] == eps[i] * (A[i]@x + b[i])
            #         for i in range(w)
            #     ] + [ y >= 0 ] + [ 0 == a_new@x + b_new ]
            # objective = cp.Minimize(sum( 10. if i==j else y[i] for i in range(w) ))
            # problem = cp.Problem( objective, constraints )
            # problem.solve(solver=cp.ECOS)
            # print(y.value)
            # k = y.value.argmin()
            # print(k)
            # # other_endpoint = np.linalg.solve(A[k],-b[k])
            # assert not (j==k)
            # #
            # # ~~~ Verify correctness
            # x = cp.Variable(dim)
            # constraints = [
            #         eps[i] * (A[i]@x + b[i]) >= 0
            #         for i in range(w)
            #     ] + [ A[k]@x + b[k] ==0 ] + [ 0 == a_new@x + b_new ]
            # problem = cp.Problem( cp.Minimize(cp.norm(x)**2), constraints )
            # problem.solve(solver=cp.ECOS)
            # correct = not (problem.status == cp.INFEASIBLE)
            # assert correct
            # other_endpoint = x.value
            # assert not (one_endpoint==other_endpoint)


#
# ~~~ Call `solve_for_where_shallow_relu_net_is_zero` but, also, plot the results
def minimalist_heatmap_where_relu_net_is_not_smooth( model, x_test, verbose=True, tol=1e-5, show=True, res=701, color="red", linewidth=2, figax=None, method="highs" ):
    #
    # ~~~ Assume that x_test==column_stack([X.flatten(),Y.flatten()]) where X,Y=meshgrid(x,y) where len(x)==len(y)==res
    res = math.sqrt(len(x_test))
    xlim = [ x_test[:,0].min().item() , x_test[:,0].max().item() ]
    ylim = [ x_test[:,1].min().item() , x_test[:,1].max().item() ]
    assert res == int(res)
    res = int(res)
    with torch.no_grad():
        Z = model(x_test).reshape(res,res)
    fig, ax = plt.subplots(figsize=(12,6)) if figax is None else figax
    heatmap = ax.imshow(
            Z, 
            extent = [ xlim[0], xlim[1], ylim[0], ylim[1] ],
            origin = "lower",
            cmap = "viridis",
            aspect = "auto"
        )    
    cbar = plt.colorbar(heatmap,ax=ax)
    cbar.set_label( "Value of the Network", rotation=270, labelpad=15 )
    #
    # ~~~ Add the lines of discontinuity introduced by the first hidden layer
    A = model[0].weight.data.cpu().double().numpy()
    b = model[0].bias.data.cpu().double().numpy()
    w,_ = A.shape
    x_line = np.linspace( xlim[0], xlim[1], 1001 )
    for i in range(w):
        y_line = -(A[i,0] * x_line + b[i]) / A[i,1]
        ax.plot(
                x_line,
                y_line,
                color = color,
                linewidth = linewidth
            )
    #
    # ~~~ If the model has two hidden layers, then we need to add the segments of discontinuity created by the second hidden layer
    if len(model)>2:
        #
        # ~~~ First, derive expression for the polygons formed by the lines that we already plotted
        sign_patterns = derive_signs_for_linear_constraints_of_a_shallow_net( A, b, verbose=verbose, tol=None )
        C = model[2].weight.data.cpu().double().numpy()
        D = model[2].bias.data.cpu().double().numpy()
        with support_for_progress_bars():
            iterator = tqdm( zip(C,D), ascii=" >=", total=len(D), desc="Finding segments of discontinuity from the second layer" ) if verbose else zip(C,D)
            #
            # ~~~ For each hidden unit of the second layer, solve for where the input to that unit is zero on each polygon
            for c,d in iterator:
                segments = solve_for_where_shallow_relu_net_is_zero( A, b, c, d, sign_patterns, verbose=verbose, tol=tol, method=method )
                #
                # ~~~ Plot the segments of discontinuity introduced by this particular hidden unit
                univar_grid = np.linspace(0,1,15)   # ~~~ a discretization of [0,1]
                for segment in segments:
                    start, end = segment
                    bivar_grid = start + np.outer( univar_grid, (end-start) )   # ~~~ a discretization of the line segment start+t*(end-start) for t in [0,1]
                    with torch.no_grad():
                        bivar_grid_torch = torch.from_numpy(bivar_grid).to(torch.get_default_dtype())
                        input_to_this_hidden_unit_on_segment = c@model[1](model[0](bivar_grid_torch)).T.numpy() + d
                    if np.max(np.abs(input_to_this_hidden_unit_on_segment)) < 1e-5:  # ~~~ one last attempt to catch incorrect resutls
                        ax.plot(
                                bivar_grid[:,0],
                                bivar_grid[:,1],
                                color = color,
                                linewidth = linewidth,
                                label = "_nolegend_"
                            )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Heatmap of the Output of a ReLU Network, Showing Seams Between Linear Pieces")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig, ax

#
# ~~~ Call `solve_for_where_shallow_relu_net_is_zero` but, also, plot the results
def plot_where_relu_net_is_not_smooth( model, xlim=[-8,8], ylim=[-8,8], verbose=True, tol=1e-5, method="highs", surface=True, show=True, res=501 ):
    #
    # ~~~ Instantiate the plot, and add at first the easy lines of discontinuity
    if surface:
        surface_or_heatmap = cell_viz( model, xlim, ylim, show=False, res=res )
    else:
        surface_or_heatmap = cell_viz( model, xlim, ylim, show=False, res=res, graph_object="heatmap" )
    lines = []
    A = model[0].weight.data.cpu().double().numpy()
    b = model[0].bias.data.cpu().double().numpy()
    w,_ = A.shape
    x_line = np.linspace( xlim[0], xlim[1], 1001 )
    for i in range(w):
        y_line = -(A[i,0] * x_line + b[i]) / A[i,1]
        with torch.no_grad():
            z_line = model( torch.from_numpy(np.column_stack([x_line, y_line])).to(torch.get_default_dtype()) ).squeeze()
        if surface:
            lines.append(go.Scatter3d(
                    x = x_line,
                    y = y_line,
                    z = z_line.numpy() + ( surface_or_heatmap["z"].max() - surface_or_heatmap["z"].min() )/100,
                    mode = "lines",
                    line = dict(color="red", width=2),
                    showlegend = False
                ))
        else:
            lines.append(go.Scatter(
                    x = x_line,
                    y = y_line,
                    mode = "lines",
                    line = dict(color="red", width=2),
                    showlegend = False
                ))
    #
    # ~~~ If the model is a shallow network, then we're done; those are the only lines of discontinuity
    if len(model)==2:
        fig = go.Figure( data = [surface_or_heatmap]+lines )
        if surface:
            fig.update_layout(
                scene = dict(
                    xaxis = dict( title='x', range=xlim ),
                    yaxis = dict( title='y', range=ylim ),
                    zaxis = dict( title='z', range=buffer([ surface_or_heatmap["z"].min(), surface_or_heatmap["z"].max() ]) )
                ))
        else:
            fig.update_layout(
                    xaxis = dict( title='x', range=xlim ),
                    yaxis = dict( title='y', range=ylim )
                )
        if show:
            fig.show()
            return None
        else:
            return fig
    #
    # ~~~ If the model has two hidden layers, then we need to add the segments of discontinuity created by the second hidden layer
    sign_patterns = derive_signs_for_linear_constraints_of_a_shallow_net( A, b, verbose=verbose, tol=None )
    C = model[2].weight.data.cpu().double().numpy()
    D = model[2].bias.data.cpu().double().numpy()
    with support_for_progress_bars():
        iterator = tqdm( zip(C,D), ascii=" >=", total=len(D), desc="Finding segments of discontinuity from the second layer" ) if verbose else zip(C,D)
        for c,d in iterator:
            segments = solve_for_where_shallow_relu_net_is_zero( A, b, c, d, sign_patterns, verbose=verbose, tol=tol, method=method )
            univar_grid = np.linspace(0,1,15)
            for segment in segments:
                start, end = segment
                bivar_grid = start + np.outer( univar_grid, (end-start) )
                with torch.no_grad():
                    bivar_grid_torch = torch.from_numpy(bivar_grid).to(torch.get_default_dtype())
                    height_of_surface_on_segment = model(bivar_grid_torch).squeeze()
                    input_to_this_hidden_unit_on_segment = c@model[1](model[0](bivar_grid_torch)).T.numpy() + d
                if np.max(np.abs(input_to_this_hidden_unit_on_segment)) < 1e-5:  # ~~~ one last attempt to catch incorrect resutls
                    if surface:
                        lines.append(go.Scatter3d(
                                x = bivar_grid[:,0],
                                y = bivar_grid[:,1],
                                z = height_of_surface_on_segment.numpy() + ( surface_or_heatmap["z"].max() - surface_or_heatmap["z"].min() )/100,
                                mode = "lines",
                                line = dict(color="red", width=2),
                                showlegend = False
                            ))
                    else:
                        lines.append(go.Scatter(
                            x = bivar_grid[:,0],
                            y = bivar_grid[:,1],
                            mode = "lines",
                            line = dict(color="red", width=2),
                            showlegend = False
                        ))
    fig = go.Figure( data = [surface_or_heatmap]+lines )
    if surface:
        fig.update_layout(
            scene = dict(
                xaxis = dict( title='x', range=xlim ),
                yaxis = dict( title='y', range=ylim ),
                zaxis = dict( title='z', range=buffer([ surface_or_heatmap["z"].min(), surface_or_heatmap["z"].max() ]) )
            ))
    else:
        fig.update_layout(
                xaxis = dict( title='x', range=xlim ),
                yaxis = dict( title='y', range=ylim )
            )
    if show:
        fig.show()
        return None
    else:
        return fig

#
# ~~~ Call `solve_for_where_shallow_relu_net_is_zero` but, also, plot the results
def plot_where_shallow_relu_net_is_zero( model, xlim=[-8,8], ylim=[-8,8], verbose=True, tol=1e-5, method="highs" ):
    A = model[0].weight.data.cpu().double().numpy()
    b = model[0].bias.data.cpu().double().numpy()
    sign_patterns = derive_signs_for_linear_constraints_of_a_shallow_net( A, b, verbose=verbose, tol=tol, desc="Deriving Linear Pieces of the First Layer" )
    c = model[-1].weight.data.cpu().double().numpy()
    d = model[-1].bias.data.cpu().double().numpy()
    segments = solve_for_where_shallow_relu_net_is_zero( A, b, c, d, sign_patterns, verbose=verbose, tol=tol, method=method )
    surface = cell_viz( model, xlim, ylim, show=False, res=1001 )
    lines = []
    univar_grid = np.linspace(0,1,15)
    for segment in segments:
        a, b = segment
        bivar_grid = a + np.outer( univar_grid, (b-a) )
        with torch.no_grad():
            height_of_surface_on_segment = model(
                    torch.from_numpy(bivar_grid).to(torch.get_default_dtype())
                ).squeeze()
        if height_of_surface_on_segment.abs().max() < 1e-5:  # ~~~ one last attempt to catch incorrect resutls
            lines.append(go.Scatter3d(
                    x = bivar_grid[:,0],
                    y = bivar_grid[:,1],
                    z = height_of_surface_on_segment.numpy(),
                    mode = "lines",
                    line = dict(color="red", width=2),
                    showlegend = False
                ))
    fig = go.Figure( data = [surface]+lines )
    fig.update_layout(
        scene=dict(
            xaxis = dict( title='x', range=xlim ),
            yaxis = dict( title='y', range=ylim ),
            zaxis = dict( title='z', range=buffer([ surface["z"].min(), surface["z"].max() ]) )
        )
    )
    fig.show()

if __name__ == "__main__":
    from torch import nn
    _ = torch.manual_seed(2024)
    w = 8
    d = 2
    model = nn.Sequential(
            nn.Linear(d,w),
            nn.ReLU(),
            nn.Linear(w,1)
        )
    plot_where_shallow_relu_net_is_zero( model, method="highs" )
    x_test = cartesian_product( torch.linspace(-8,8,501), torch.linspace(-8,8,501) )
    minimalist_heatmap_where_relu_net_is_not_smooth( model, x_test, tol=1e-5, method="highs" )
    model = nn.Sequential(
            nn.Linear(d,w),
            nn.ReLU(),
            nn.Linear(w,w),
            nn.ReLU(),
            nn.Linear(w,1)
        )
    plot_where_relu_net_is_not_smooth( model, xlim=[-8,8], ylim=[-8,8], method="highs" )
    minimalist_heatmap_where_relu_net_is_not_smooth( model, x_test, tol=1e-5, method="highs" )

#