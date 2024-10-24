
import torch

### ~~~
## ~~~ Compute the vectors a_j and b_j for which we demand the constraint |a_j^Tz| \leq |b_j^Tz|
### ~~~

def build_b_j(x,j):
    #
    # ~~~ Use 1-indexing, assuming that the given j is in zero-indexing to begin with
    assert len(x)%2==0
    k = len(x)//2
    assert j >= 0 and j <= k-1
    j += 1
    x_2jp2 = x[2*j+2-1]
    x_2jp1 = x[2*j+1-1]
    x_2j = x[2*j-1]
    x_2jm1 = x[2*j-1-1]
    #
    # ~~~ Compute the non-zero coordinates of the vector b_j
    # Precompute common factor
    common_factor = (x_2jp1 - x_2j) / 2
    coeff_2jp2 = common_factor / (x_2jp2 - x_2jp1)
    coeff_2jp1 = -common_factor / (x_2jp2 - x_2jp1)
    coeff_2j = -common_factor / (x_2j - x_2jm1)
    coeff_2jm1 = common_factor / (x_2j - x_2jm1)
    #
    # ~~~ Assign the computed coefficients to the non-zero positions in the vector a_j
    b_j = torch.zeros_like(x)
    b_j[2*j+2-1] = coeff_2jp2
    b_j[2*j+1-1] = coeff_2jp1
    b_j[2*j-1] = coeff_2j
    b_j[2*j-1-1] = coeff_2jm1
    return b_j

def build_a_j(x,j):
    #
    # ~~~ Use 1-indexing, assuming that the given j is in zero-indexing to begin with
    assert len(x)%2==0
    k = len(x)//2
    assert j >= 0 and j <= k-1
    j += 1
    x_2jp2 =  x[2*j+2-1]    # ~~~ x_{2*j+2}
    x_2jp1 =  x[2*j+1-1]    # ~~~ x_{2*j+1}
    x_2j   =  x[2*j-1]      # ~~~ x_{2*j}
    x_2jm1 =  x[2*j-1-1]    # ~~~ x_{2*j-1}
    c_j    =  (x_2j   + x_2jp1)/2   # ~~~ c_j = (x_{2j} + x_{2j+1})/2, the midpoint of the interval where a break point is allowed
    d_j    =  (x_2j   - x_2jm1)     # ~~~ d_j = x_{2j} - x_{2j-1}, the length of one of the intervals in which no break point is allowed
    d_jp1  =  (x_2jp2 - x_2jp1)     # ~~~ d_{j+1} = x_{2j+2} - x_{2j+1}, the length of one of the intervals in which no break point is allowed
    m_j    =  (x_2j   + x_2jm1)/2   # ~~~ m_j = (x_{2j} + x_{2j-1})/2, the midpoint of one of the intervals in which no break point is allowed
    m_jp1  =  (x_2jp2 + x_2jp1)/2   # ~~~ m_{j+1} = (x_{2j+2} + x_{2j+1})/2, the midpoint of one of the intervals in which no break point is allowed
    #
    # ~~~ Compute the non-zero coordinates of the vector a_j
    coeff_2jp2 = (
        (x_2jp2 + x_2jp1) / (2 * (x_2jp2 - x_2jp1)) 
        - 1/2 
        - (x_2j + x_2jp1) / (2 * (x_2jp2 - x_2jp1))
    )
    coeff_2jp1 = (
        -(x_2jp2 + x_2jp1) / (2 * (x_2jp2 - x_2jp1)) 
        + 1/2 
        + (x_2j + x_2jp1) / (2 * (x_2jp2 - x_2jp1))
    )
    coeff_2j = (
        -(x_2j + x_2jm1) / (2 * (x_2j - x_2jm1)) 
        + 1/2 
        + (x_2j + x_2jm1) / (2 * (x_2j - x_2jm1))
    )
    coeff_2jm1 = (
        (x_2j + x_2jm1) / (2 * (x_2j - x_2jm1)) 
        - 1/2 
        - (x_2j + x_2jm1) / (2 * (x_2j - x_2jm1))
    )
    #
    # ~~~ Assign the computed coefficients to the non-zero positions in the vector a_j
    a_j = torch.zeros_like(x)
    a_j[2*j+2-1] = coeff_2jp2
    a_j[2*j+1-1] = coeff_2jp1
    a_j[2*j-1] = coeff_2j
    a_j[2*j-1-1] = coeff_2jm1
    return a_j


if __name__ == "__main__":
    #
    # ~~~ Config
    torch.manual_seed(2024)
    k = 15
    m =  2*k
    f = lambda x: torch.sin(2*torch.pi*x)
    x_train = torch.linspace(-1,1,m)
    y_train = f(x_train)
    v = spline(x_train)
    x_test = torch.linspace(-1,1,1001)
    y_test = f(x_test)
    # points_with_curves( x=x_train,  y=y_train, curves=(v,f) )