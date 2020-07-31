import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcParams.update({'font.size': 18})


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def get_natural_cubic_spline_model(x, y, minval=None, maxval=None, n_knots=None, knots=None):
    """
    Get a natural cubic spline model for the data.

    For the knots, give (a) `knots` (as an array) or (b) minval, maxval and n_knots.

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    x: np.array of float
        The input data
    y: np.array of float
        The outpur data
    minval: float 
        Minimum of interval containing the knots.
    maxval: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.

    Returns
    --------
    model: a model object
        The returned model will have following method:
        - predict(x):
            x is a numpy array. This will return the predicted y-values.
    """

    if knots:
        spline = NaturalCubicSpline(knots=knots)
    else:
        spline = NaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)

    p = Pipeline([
        ('nat_cubic', spline),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    p.fit(x, y)

    return p


class AbstractSpline(BaseEstimator, TransformerMixin):
    """Base class for all spline basis expansions."""

    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
        if knots is None:
            if not n_knots:
                n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self


class NaturalCubicSpline(AbstractSpline):
    """Apply a natural cubic basis expansion to an array.
    The features created with this basis expansion can be used to fit a
    piecewise cubic function under the constraint that the fitted curve is
    linear *outside* the range of the knots..  The fitted curve is continuously
    differentiable to the second order at all of the knots.
    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots.
      - By specifying the cutpoints directly.  

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.
    Parameters
    ----------
    min: float 
        Minimum of interval containing the knots.
    max: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.
    """

    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError: # For arrays with only one element
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            def ppart(t): return np.maximum(0, t)

            def cube(t): return t*t*t
            numerator = (cube(ppart(x - self.knots[knot_idx]))
                         - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
        return X_spl


# The number of knots can be used to control the amount of smoothness

# deg_center = [0, 110, 95, 95, 0, 115, 125, 0, 75, 80, 80, 90, 95, 105, 120, 125, 140, 0]


data = np.load("angles_7e-6.npy")

print(data.shape)

# plt.figure(figsize=(12,8))
deg_crossings = {}

for feed in tqdm([1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19]):

    clean_idx = ~np.isnan(data[feed])
    deg = data[0][clean_idx]
    signal = data[feed][clean_idx]

    def gen_log(x, A, K, B, Q, mu, M, C):
        return A + (K - A)/(C + Q*np.exp(-B*(x-M)))**(1.0/mu)

    def fsigmoid(x, a, b):
        return 1.0 / (1.0 + np.exp(-a*(x-b)))
        # return a*x + b

    # deg_new = (deg-120.0)/60.0

    # popt, pcov = curve_fit(gen_log, deg_new, 1- signal, bounds=([-100, -100, -100, -100, -100, -10, -100], [100, 100, 100, 100, 100, 100, 100]), p0 = [0, 1, 3.0, 0.5, 1.0, 0.0, 1.0])
    # print(popt, pcov)
    signal_sorted = signal[np.argsort(deg)]
    deg_sorted = deg[np.argsort(deg)]


    model_6 = get_natural_cubic_spline_model(deg_sorted, signal_sorted, minval=min(deg), maxval=160, n_knots=6)
    model_15 = get_natural_cubic_spline_model(deg_sorted, signal_sorted, minval=min(deg), maxval=160, n_knots=15)
    model_20 = get_natural_cubic_spline_model(deg_sorted, signal_sorted, minval=min(deg), maxval=160, n_knots=20)
    model_30 = get_natural_cubic_spline_model(deg_sorted, signal_sorted, minval=min(deg), maxval=160, n_knots=30)
    model_40 = get_natural_cubic_spline_model(deg_sorted, signal_sorted, minval=min(deg), maxval=160, n_knots=40)
    model_60 = get_natural_cubic_spline_model(deg_sorted, signal_sorted, minval=min(deg), maxval=160, n_knots=60)
    y_est_6 = model_6.predict(deg_sorted)
    y_est_15 = model_15.predict(deg_sorted)
    y_est_20 = model_20.predict(deg_sorted)
    y_est_30 = model_30.predict(deg_sorted)
    y_est_40 = model_40.predict(deg_sorted)
    y_est_60 = model_60.predict(deg_sorted)

    deg_crossing_idx = np.argwhere(y_est_15 < 0.97)[0][0]

    deg_crossings[feed] = deg_sorted[deg_crossing_idx]

    # spl = UnivariateSpline(deg_sorted, signal_sorted, k=5)

    plt.figure(figsize=(12, 8))
    plt.axvline(x=deg_sorted[deg_crossing_idx], c="y", ls="--")
    plt.axhline(y=y_est_15[deg_crossing_idx], c="y", ls="--")
    plt.scatter(deg, signal, s=0.2, label="FEED=%d" % feed)
    # plt.plot(deg_sorted, y_est_6, c="k")
    plt.plot(deg_sorted, y_est_15)
    # plt.plot(deg_sorted, y_est_30, c="r")
    # plt.plot(deg_sorted, y_est_60, c="y")
    # plt.legend()
    # plt.plot(deg_sorted, y_est_15, label="FEED=%d" % feed)
    plt.xlabel("Vane angle [Degrees]")
    plt.ylabel("Normalized TOD")
    plt.title("All of 2020-04 | Feed %d | Offset = 7e-6 MJD" % feed)
    plt.tight_layout()
    plt.savefig("power_angle_feed%d.png" % feed, bbox_inches="tight")
    plt.close()
    plt.clf()
# plt.legend(loc=1)
# plt.tight_layout()
# plt.savefig("power_angle_splines_15.png", bbox_inches="tight")
# plt.close()
# plt.clf()
# print(deg_crossings)