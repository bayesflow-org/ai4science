"""Minimal univariate MMAR simulator used by the tutorial."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW = 256
CASCADE_BLOCK = 32
PARAMETER_NAMES = ("mu", "sigma_bar", "q", "nu")
PARAMETER_LABELS = (r"$\mu$", r"$\bar\sigma$", r"$q$", r"$\nu$")


@dataclass(frozen=True)
class Prior:
    """Prior bounds for the four MMAR parameters."""

    mu_loc: float = 0.0
    mu_sd: float = 0.0015
    sigma_low: float = 0.005
    sigma_high: float = 0.025
    q_low: float = 0.52
    q_high: float = 0.90
    nu_low: float = 2.05
    nu_high: float = 12.0


PRIOR = Prior()


def prior_table(prior=PRIOR):
    """Return a readable table of parameter priors and roles."""
    return pd.DataFrame(
        {
            "prior": [
                f"Normal({prior.mu_loc:g}, {prior.mu_sd:g}²)",
                f"Uniform({prior.sigma_low:g}, {prior.sigma_high:g})",
                f"Uniform({prior.q_low:g}, {prior.q_high:g})",
                (
                    "2 + exp(Uniform("
                    f"log({prior.nu_low - 2:.4g}), "
                    f"log({prior.nu_high - 2:g})))"
                ),
            ],
            "meaning": [
                "Daily drift",
                "Baseline RMS return scale",
                "Cascade contrast and volatility clustering",
                "Student-t degrees of freedom and tail thickness",
            ],
        },
        index=PARAMETER_NAMES,
    )


def _draw_parameters(n_simulations, rng, prior):
    sigma = rng.uniform(prior.sigma_low, prior.sigma_high, n_simulations)
    q = rng.uniform(prior.q_low, prior.q_high, n_simulations)
    log_nu_minus_two = rng.uniform(
        np.log(prior.nu_low - 2),
        np.log(prior.nu_high - 2),
        n_simulations,
    )
    nu = 2 + np.exp(log_nu_minus_two)
    mu = rng.normal(prior.mu_loc, prior.mu_sd, n_simulations)
    return np.column_stack((mu, sigma, q, nu))


def stack_samples(samples):
    """Stack named posterior draws in the simulator's parameter order."""
    return np.concatenate([samples[name] for name in PARAMETER_NAMES], axis=-1)


def _cascade(q, rng):
    """Generate eight 32-day volatility blocks with a random phase."""
    weights = np.ones((len(q), 1))
    high = 2 * q[:, None]
    low = 2 * (1 - q[:, None])

    for _ in range(int(np.log2(WINDOW // CASCADE_BLOCK))):
        high_goes_left = rng.random(weights.shape) < 0.5
        left = weights * np.where(high_goes_left, high, low)
        right = weights * np.where(high_goes_left, low, high)
        weights = np.stack((left, right), axis=-1).reshape(len(q), -1)

    weights /= weights.mean(axis=1, keepdims=True)
    weights = np.repeat(weights, CASCADE_BLOCK, axis=1)
    offsets = rng.integers(0, WINDOW, size=(len(q), 1))
    indices = (np.arange(WINDOW)[None, :] + offsets) % WINDOW
    return np.take_along_axis(weights, indices, axis=1)


def _simulate_from_parameters(parameters, rng):
    """Simulate one 256-day return path per parameter vector."""
    mu, sigma, q, nu = parameters.T
    returns = np.empty((len(parameters), WINDOW))
    pending = np.arange(len(parameters))

    for _ in range(100):
        trading_time = _cascade(q[pending], rng)
        df = nu[pending, None]
        innovations = rng.standard_t(
            df, size=(len(pending), WINDOW)
        ) * np.sqrt((df - 2) / df)
        proposed = (
            mu[pending, None]
            + sigma[pending, None] * np.sqrt(trading_time) * innovations
        )
        returns[pending] = proposed
        invalid = (~np.isfinite(proposed).all(axis=1)) | (proposed <= -1).any(
            axis=1
        )
        pending = pending[invalid]
        if not len(pending):
            return {
                "mu": mu[:, None],
                "sigma_bar": sigma[:, None],
                "q": q[:, None],
                "nu": nu[:, None],
                "returns": returns,
            }

    raise RuntimeError("Could not draw valid return paths in 100 attempts.")


def simulate(n_simulations, rng=None, prior=PRIOR):
    """Draw parameters from the prior and simulate matching return paths."""
    if rng is None:
        rng = np.random.default_rng()
    parameters = _draw_parameters(n_simulations, rng, prior)
    return _simulate_from_parameters(parameters, rng)


def q_scenarios(q_values, n_paths=4, seed=20260906):
    """Simulate matched paths while varying only the cascade parameter q."""
    return {
        float(q): 100
        * _simulate_from_parameters(
            np.tile([0.0, 0.015, q, 4.0], (n_paths, 1)),
            np.random.default_rng(seed),
        )["returns"]
        for q in q_values
    }


def posterior_resimulations(posterior, rng=None):
    """Draw fresh return paths for each batch of posterior parameters."""
    if rng is None:
        rng = np.random.default_rng()
    return np.stack(
        [
            _simulate_from_parameters(draws, rng)["returns"]
            for draws in posterior
        ]
    )
