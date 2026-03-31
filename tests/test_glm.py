"""Tests for GLM implementation."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glm import GLM, GLMResult
from families import Gaussian, Binomial, Poisson


RNG = np.random.default_rng(42)
N = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_data(n=N, p=3, noise=0.5):
    X = RNG.standard_normal((n, p))
    true_coef = np.array([1.0, 0.5, -1.0, 0.2])  # includes intercept
    mu = np.hstack([np.ones((n, 1)), X]) @ true_coef
    y = mu + RNG.normal(0, noise, n)
    return X, y, true_coef


def _binomial_data(n=N, p=3):
    X = RNG.standard_normal((n, p))
    true_coef = np.array([0.5, 1.0, -0.5, 0.3])
    eta = np.hstack([np.ones((n, 1)), X]) @ true_coef
    prob = 1 / (1 + np.exp(-eta))
    y = RNG.binomial(1, prob).astype(float)
    return X, y, true_coef


def _poisson_data(n=N, p=3):
    X = RNG.standard_normal((n, p))
    true_coef = np.array([0.3, 0.5, -0.3, 0.2])
    eta = np.hstack([np.ones((n, 1)), X]) @ true_coef
    lam = np.exp(eta)
    y = RNG.poisson(lam).astype(float)
    return X, y, true_coef


# ---------------------------------------------------------------------------
# Family tests
# ---------------------------------------------------------------------------

class TestGaussianFamily:
    fam = Gaussian()

    def test_link_inverse_are_identity(self):
        mu = np.array([0.0, 1.0, -2.0, 3.5])
        np.testing.assert_array_equal(self.fam.link(mu), mu)
        np.testing.assert_array_equal(self.fam.link_inv(mu), mu)

    def test_link_deriv_is_one(self):
        mu = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(self.fam.link_deriv(mu), np.ones(3))

    def test_variance_is_one(self):
        mu = np.linspace(0.1, 5.0, 10)
        np.testing.assert_array_equal(self.fam.variance(mu), np.ones(10))

    def test_deviance_zero_at_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0])
        assert self.fam.deviance(y, y) == pytest.approx(0.0)


class TestBinomialFamily:
    fam = Binomial()

    def test_link_inv_range(self):
        eta = np.linspace(-5, 5, 100)
        mu = self.fam.link_inv(eta)
        assert np.all(mu > 0) and np.all(mu < 1)

    def test_link_roundtrip(self):
        mu = np.linspace(0.05, 0.95, 20)
        np.testing.assert_allclose(self.fam.link_inv(self.fam.link(mu)), mu)

    def test_deviance_nonneg(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        mu = np.array([0.3, 0.7, 0.4, 0.8])
        assert self.fam.deviance(y, mu) >= 0

    def test_deviance_zero_at_perfect_fit(self):
        y = np.array([0.0, 1.0, 0.0])
        mu = np.clip(y, 1e-8, 1 - 1e-8)
        assert self.fam.deviance(y, mu) == pytest.approx(0.0, abs=1e-6)


class TestPoissonFamily:
    fam = Poisson()

    def test_link_inv_positive(self):
        eta = np.linspace(-5, 5, 100)
        assert np.all(self.fam.link_inv(eta) > 0)

    def test_link_roundtrip(self):
        mu = np.array([0.5, 1.0, 2.0, 5.0])
        np.testing.assert_allclose(self.fam.link_inv(self.fam.link(mu)), mu)

    def test_variance_equals_mean(self):
        mu = np.array([1.0, 2.0, 5.0])
        np.testing.assert_array_equal(self.fam.variance(mu), mu)

    def test_deviance_nonneg(self):
        y = np.array([0.0, 1.0, 3.0, 5.0])
        mu = np.array([0.5, 1.5, 2.5, 4.5])
        assert self.fam.deviance(y, mu) >= 0


# ---------------------------------------------------------------------------
# GLM fitting tests
# ---------------------------------------------------------------------------

class TestGLMGaussian:
    def test_converges(self):
        X, y, _ = _gaussian_data()
        result = GLM(family=Gaussian()).fit(X, y)
        assert result.converged

    def test_coef_close_to_truth(self):
        X, y, true_coef = _gaussian_data()
        result = GLM(family=Gaussian()).fit(X, y)
        np.testing.assert_allclose(result.coef, true_coef, atol=0.15)

    def test_predict_shape(self):
        X, y, _ = _gaussian_data()
        result = GLM(family=Gaussian()).fit(X, y)
        preds = result.predict(X)
        assert preds.shape == (N,)

    def test_no_intercept(self):
        X = RNG.standard_normal((200, 2))
        y = X @ np.array([1.0, -0.5]) + RNG.normal(0, 0.1, 200)
        result = GLM(family=Gaussian(), fit_intercept=False).fit(X, y)
        assert result.coef.shape == (2,)
        np.testing.assert_allclose(result.coef, [1.0, -0.5], atol=0.1)

    def test_1d_x(self):
        x = RNG.standard_normal(200)
        y = 2.0 * x + RNG.normal(0, 0.1, 200)
        result = GLM(family=Gaussian(), fit_intercept=False).fit(x, y)
        assert result.coef[0] == pytest.approx(2.0, abs=0.1)


class TestGLMBinomial:
    def test_converges(self):
        X, y, _ = _binomial_data()
        result = GLM(family=Binomial()).fit(X, y)
        assert result.converged

    def test_coef_close_to_truth(self):
        X, y, true_coef = _binomial_data()
        result = GLM(family=Binomial()).fit(X, y)
        np.testing.assert_allclose(result.coef, true_coef, atol=0.3)

    def test_predictions_in_01(self):
        X, y, _ = _binomial_data()
        result = GLM(family=Binomial()).fit(X, y)
        preds = result.predict(X)
        assert np.all(preds > 0) and np.all(preds < 1)

    def test_link_scale_predictions(self):
        X, y, _ = _binomial_data()
        result = GLM(family=Binomial()).fit(X, y)
        eta = result.predict(X, transform=False)
        assert eta.shape == (N,)


class TestGLMPoisson:
    def test_converges(self):
        X, y, _ = _poisson_data()
        result = GLM(family=Poisson()).fit(X, y)
        assert result.converged

    def test_coef_close_to_truth(self):
        X, y, true_coef = _poisson_data()
        result = GLM(family=Poisson()).fit(X, y)
        np.testing.assert_allclose(result.coef, true_coef, atol=0.15)

    def test_predictions_positive(self):
        X, y, _ = _poisson_data()
        result = GLM(family=Poisson()).fit(X, y)
        preds = result.predict(X)
        assert np.all(preds > 0)


class TestGLMResult:
    def test_repr(self):
        X, y, _ = _gaussian_data()
        result = GLM().fit(X, y)
        r = repr(result)
        assert "Gaussian" in r
        assert "converged=True" in r

    def test_deviance_nonneg(self):
        X, y, _ = _binomial_data()
        result = GLM(family=Binomial()).fit(X, y)
        assert result.deviance >= 0
