
import numpy as np
import cvxpy as cp
from near_optimal.quadratic_univar import DualSpline

class SemidefSpline(DualSpline):
    
    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
    
    def fit(self):
        y = self.y
        m = len(y)
        A = np.eye(m+1)
        A[m,m] = 0
        A[:m,m] = -y
        A[m,:m] = -y
        aa = [ np.append(a,0) for a in self.a ]
        bb = [ np.append(b,0) for b in self.b ]
        Z = cp.Variable( (m+1,m+1), PSD=True )
        constraints = [ Z[m,m] == 1 ] + [
                a@Z@a <= b@Z@b
                for a,b in zip(aa,bb)
            ]
        objective = cp.Minimize(cp.trace(A@Z))
        problem = cp.Problem( objective, constraints )
        problem.solve()
        print(problem.objective.value)

if __name__ == "__main__":
    from near_optimal.quadratic_univar import x_train, y_train, x_test, y_test
    v = SemidefSpline( x_train, y_train )
    v.fit() # ~~~ returns a negative value, hence appears to not work