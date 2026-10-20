"""
Implement MARS, as a point of comparison.
When I google it, the package https://github.com/scikit-learn-contrib/py-earth comes up.
However, this package has not been maintained for years and is at this point not installable without using obsolete versions of python
Chatgpt pointed out to me that there is a user-maintained updated version of the package https://github.com/felixgwilliams/py-earth2.
That iinstalled fine. However, the demo code from the README just straight up threw an error.
chatgpt then wrote the following I told it to try to come up with a work-around that dodges the error in the source code of the latter package.
"""

# ------ AI generated ------
import numpy as np
from pyearth import Earth
import pyearth._basis as _basis


class _NumpyCompat:
    def __getattr__(self, name):
        return getattr(np, name)

    def empty(self, shape, dtype=float, *args, **kwargs):
        if dtype is int:
            dtype = np.int64
        return np.empty(shape, dtype, *args, **kwargs)


def fit_UARS(x_train, y_train, verbose=True):
    """Fit a univariate MARS model and return its prediction function."""
    model = Earth(max_degree=1, max_terms=20)

    original_numpy = _basis.np
    _basis.np = _NumpyCompat()
    try:
        model.fit(
            np.asarray(x_train, dtype=float).reshape(-1, 1),
            np.asarray(y_train, dtype=float).ravel(),
        )
    finally:
        _basis.np = original_numpy

    if verbose:
        print(model.summary())
    
    def UARS(x):
        """Return a flat vector of predictions at x."""
        return model.predict(
            np.asarray(x, dtype=float).reshape(-1, 1)
        ).ravel()

    return UARS

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2, 200)
    y = np.abs(x - 0.3) + 0.1 * rng.standard_normal(x.size)
    UARS = fit_UARS(x,y)

    xx = np.linspace(-2, 2, 500)
    yy = UARS(xx)
    plt.scatter(x, y, s=12, alpha=0.5)
    plt.plot(xx, yy, color="black")
    plt.show()
# --------------------------
