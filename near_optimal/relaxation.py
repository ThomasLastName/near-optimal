"""
In this file, we try the natural semi-definite of the problem

\min_z (1/m)\|y-z\|_2^2  s.t. (z^Ta_j)^2 \leq (z^Tb)^2, j<k

Namely, we introduce Z as a stand-in for the outer product zz^T
Then, to deal with anywhere that z enters quadratically, we use z
and, to relate Z back to z, we employ the constraint Z \geq zz^T
(a semi-definite inequality) which can be forced as a semi-definite
constraint using the Schur complement.
See https://www.princeton.edu/~aaa/Public/Teaching/ORF523/ORF523_Lec12.pdf
Empirically, this doesn't work. Just sayin', we tried it.
"""

import numpy as np
import cvxpy as cp
from near_optimal.quadratic_univar import DualSpline
from quality_of_life.my_cvx_utils import Schur_complement

class SemidefSpline(DualSpline):
    
    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
    
    def fit(self):
        y = self.y.numpy()
        m = len(y)
        H_0 = np.eye(m) # ~~~ see https://www.princeton.edu/~aaa/Public/Teaching/ORF523/ORF523_Lec12.pdf
        c_0 =  -y       # ~~~ see https://www.princeton.edu/~aaa/Public/Teaching/ORF523/ORF523_Lec12.pdf
        d_0 = sum(y**2) # ~~~ see https://www.princeton.edu/~aaa/Public/Teaching/ORF523/ORF523_Lec12.pdf
        aa = self.a
        bb = self.b
        Z = cp.Variable( (m,m), PSD=True )  
        z = cp.Variable(m)
        constraints = [ Schur_complement(Z,z) ] + [ # ~~~ the first constraint is that Z \geq zz^T
                a@Z@a <= b@Z@b                      # ~~~ relaxation of the constraint (z^Ta)^2 \leq (z^Tb)^2
                for a,b in zip(aa,bb)
            ]
        objective = cp.Minimize( (cp.trace(H_0@Z) + sum(cp.multiply(c_0,z)) + d_0)/m )
        problem = cp.Problem( objective, constraints )
        problem.solve()

if __name__ == "__main__":
    from near_optimal.quadratic_univar import x_train, y_train
    v = SemidefSpline( x_train, y_train )
    _ = v.fit() # ~~~ returns a negative value, hence appears to not work
    print("")
    print(f"This approach yielded a max abs error of {(v.z - v.y).detach().abs().max().item()} whereas we know from other experiments in this library that 0.05 is about correct.")
    print("")
    