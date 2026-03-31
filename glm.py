"""Generalized Linear Model (GLM) implementation.

Fitting is done via Iteratively Reweighted Least Squares (IRLS).
"""

import numpy as np

from families import Gaussian, Binomial, Poisson  # noqa: F401 – re-exported for convenience


class GLMResult:
    """Container for a fitted GLM."""

    def __init__(self, coef, family, converged, n_iter, deviance, fit_intercept):
        self.coef = coef          # shape (p,)
        self.family = family
        self.converged = converged
        self.n_iter = n_iter
        self.deviance = deviance
        self.fit_intercept = fit_intercept

    def predict(self, X, transform=True):
        """Return predictions on the response scale (transform=True) or link scale.

        Parameters
        ----------
        X : array-like, shape (n, p)
            Feature matrix (without intercept column).
        transform : bool
            If True, apply the inverse link to return predictions on the
            response scale. If False, return linear predictor values.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        eta = X @ self.coef
        return self.family.link_inv(eta) if transform else eta

    def __repr__(self):
        return (
            f"GLMResult("
            f"family={self.family.__class__.__name__}, "
            f"converged={self.converged}, "
            f"n_iter={self.n_iter}, "
            f"deviance={self.deviance:.6g})"
        )


class GLM:
    """Generalized Linear Model fitted by IRLS.

    Parameters
    ----------
    family : Family
        Distribution family (Gaussian, Binomial, or Poisson). Defaults to Gaussian.
    fit_intercept : bool
        Whether to prepend a column of ones to X.
    max_iter : int
        Maximum number of IRLS iterations.
    tol : float
        Convergence tolerance on the relative change in deviance.

    Examples
    --------
    Logistic regression:

    >>> import numpy as np
    >>> from glm import GLM
    >>> from families import Binomial
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 2))
    >>> p = 1 / (1 + np.exp(-(1 + X @ [0.5, -1.0])))
    >>> y = rng.binomial(1, p).astype(float)
    >>> model = GLM(family=Binomial())
    >>> result = model.fit(X, y)
    >>> result.coef  # approximately [1, 0.5, -1]

    Poisson regression:

    >>> from families import Poisson
    >>> lam = np.exp(0.5 + X @ [0.3, 0.7])
    >>> y = rng.poisson(lam).astype(float)
    >>> result = GLM(family=Poisson()).fit(X, y)
    """

    def __init__(self, family=None, fit_intercept=True, max_iter=100, tol=1e-8):
        self.family = family if family is not None else Gaussian()
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def _add_intercept(self, X):
        n = X.shape[0]
        return np.hstack([np.ones((n, 1)), X])

    def fit(self, X, y):
        """Fit the model to data.

        Parameters
        ----------
        X : array-like, shape (n, p)
            Feature matrix (without intercept column).
        y : array-like, shape (n,)
            Response vector.

        Returns
        -------
        GLMResult
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X[:, np.newaxis]

        if self.fit_intercept:
            X = self._add_intercept(X)

        n, p = X.shape
        family = self.family

        # Starting values
        mu = family.starting_mu(y)
        eta = family.link(mu)

        coef = np.zeros(p)
        deviance = np.inf

        converged = False
        n_iter = 0

        for n_iter in range(1, self.max_iter + 1):
            # Working weights and working response
            g_prime = family.link_deriv(mu)         # g'(mu)
            V = family.variance(mu)                  # V(mu)
            W = 1.0 / (g_prime ** 2 * V)            # IRLS weights
            z = eta + g_prime * (y - mu)             # adjusted dependent variable

            # Weighted least squares: solve (X^T W X) beta = X^T W z
            W_sqrt = np.sqrt(W)
            Xw = X * W_sqrt[:, np.newaxis]
            zw = z * W_sqrt

            coef, *_ = np.linalg.lstsq(Xw, zw, rcond=None)

            # Update eta and mu
            eta = X @ coef
            mu = family.link_inv(eta)

            # Check convergence
            new_deviance = family.deviance(y, mu)
            if np.isfinite(deviance) and abs(deviance - new_deviance) / (abs(deviance) + 1e-10) < self.tol:
                deviance = new_deviance
                converged = True
                break
            deviance = new_deviance

        return GLMResult(
            coef=coef,
            family=family,
            converged=converged,
            n_iter=n_iter,
            deviance=deviance,
            fit_intercept=self.fit_intercept,
        )
