"""Distribution families and link functions for GLMs."""

import numpy as np


class Family:
    """Base class for GLM distribution families."""

    def link(self, mu):
        """Apply the link function: eta = g(mu)."""
        raise NotImplementedError

    def link_inv(self, eta):
        """Apply the inverse link function: mu = g^{-1}(eta)."""
        raise NotImplementedError

    def link_deriv(self, mu):
        """Derivative of the link function: g'(mu)."""
        raise NotImplementedError

    def variance(self, mu):
        """Variance function V(mu)."""
        raise NotImplementedError

    def deviance(self, y, mu):
        """Unit deviance d(y, mu)."""
        raise NotImplementedError

    def starting_mu(self, y):
        """Starting values for mu in IRLS."""
        raise NotImplementedError


class Gaussian(Family):
    """Gaussian family with identity link.

    Suitable for continuous, normally distributed responses.
    """

    def link(self, mu):
        return mu

    def link_inv(self, eta):
        return eta

    def link_deriv(self, mu):
        return np.ones_like(mu)

    def variance(self, mu):
        return np.ones_like(mu)

    def deviance(self, y, mu):
        return np.sum((y - mu) ** 2)

    def starting_mu(self, y):
        return y.copy()


class Binomial(Family):
    """Binomial family with logit link.

    Suitable for binary or proportion responses in [0, 1].
    """

    _EPS = 1e-10

    def link(self, mu):
        mu = np.clip(mu, self._EPS, 1 - self._EPS)
        return np.log(mu / (1 - mu))

    def link_inv(self, eta):
        return 1.0 / (1.0 + np.exp(-eta))

    def link_deriv(self, mu):
        mu = np.clip(mu, self._EPS, 1 - self._EPS)
        return 1.0 / (mu * (1 - mu))

    def variance(self, mu):
        mu = np.clip(mu, self._EPS, 1 - self._EPS)
        return mu * (1 - mu)

    def deviance(self, y, mu):
        mu = np.clip(mu, self._EPS, 1 - self._EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = np.where(y > 0, y * np.log(np.where(y > 0, y / mu, 1)), 0)
            term2 = np.where(y < 1, (1 - y) * np.log(np.where(y < 1, (1 - y) / (1 - mu), 1)), 0)
        return 2 * np.sum(term1 + term2)

    def starting_mu(self, y):
        return np.clip((y + 0.5) / 2, self._EPS, 1 - self._EPS)


class Poisson(Family):
    """Poisson family with log link.

    Suitable for count data.
    """

    _EPS = 1e-10

    def link(self, mu):
        return np.log(np.maximum(mu, self._EPS))

    def link_inv(self, eta):
        return np.exp(eta)

    def link_deriv(self, mu):
        return 1.0 / np.maximum(mu, self._EPS)

    def variance(self, mu):
        return np.maximum(mu, self._EPS)

    def deviance(self, y, mu):
        mu = np.maximum(mu, self._EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(y > 0, y * np.log(np.where(y > 0, y / mu, 1)), 0)
        return 2 * np.sum(log_term - (y - mu))

    def starting_mu(self, y):
        return np.maximum(y, self._EPS)
