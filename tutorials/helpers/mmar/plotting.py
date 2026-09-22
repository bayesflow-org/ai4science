"""Plotting helpers for the minimal MMAR tutorial."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from scipy.ndimage import gaussian_filter1d

PURPLE = "#4D2F7A"
GREEN = "#285C4D"
OBSERVED = "#202124"


def configure_plot_style():
    """Apply a compact presentation style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "font.size": 13,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
        }
    )


def _max_drawdown(paths):
    wealth = np.cumprod(1 + np.asarray(paths), axis=-1)
    peaks = np.maximum.accumulate(wealth, axis=-1)
    return np.max(1 - wealth / peaks, axis=-1)


def plot_prior_predictive(prior_returns, observed_returns):
    """Compare prior simulations with one observed return window."""
    simulated = np.asarray(prior_returns)
    observed = np.asarray(observed_returns)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")

    limit = np.quantile(np.abs(simulated), 0.995)
    bins = np.linspace(-limit, limit, 70)
    axes[0].hist(
        simulated.ravel(),
        bins=bins,
        density=True,
        color="#B9A5D8",
        alpha=0.75,
        label="Prior predictive",
    )
    axes[0].hist(
        observed,
        bins=bins,
        density=True,
        histtype="step",
        color=OBSERVED,
        linewidth=2,
        label="VOO observed",
    )
    axes[0].set(
        xlabel="Daily simple return",
        ylabel="Density",
        title="Marginal returns",
    )
    axes[0].legend(frameon=False)

    drawdowns = _max_drawdown(simulated)
    observed_drawdown = _max_drawdown(observed[None, :])[0]
    axes[1].hist(
        drawdowns,
        bins=40,
        density=True,
        color="#B9A5D8",
        alpha=0.75,
    )
    axes[1].axvline(
        observed_drawdown,
        color=OBSERVED,
        linewidth=2,
        label=f"VOO observed: {100 * observed_drawdown:.1f}%",
    )
    axes[1].set(
        xlabel="Maximum drawdown",
        ylabel="Density",
        title="Path-level downside",
    )
    axes[1].legend(frameon=False)
    return fig


def plot_posterior_predictive(observed_returns, posterior_paths, dates):
    """Compare observed wealth and drawdown with posterior predictions."""
    observed = np.asarray(observed_returns)
    paths = np.asarray(posterior_paths)
    observed_wealth = np.cumprod(1 + observed)
    predictive_wealth = np.cumprod(1 + paths, axis=1)
    q04, q16, q50, q84, q96 = np.quantile(
        predictive_wealth, (0.04, 0.16, 0.5, 0.84, 0.96), axis=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].fill_between(
        dates, q04, q96, color="#D8C9F0", label="92% interval"
    )
    axes[0].fill_between(
        dates, q16, q84, color="#A98BCB", label="68% interval"
    )
    axes[0].plot(dates, q50, color=PURPLE, linewidth=2, label="Median")
    axes[0].plot(
        dates,
        observed_wealth,
        color=OBSERVED,
        linewidth=1.7,
        label="VOO observed",
    )
    axes[0].set(
        ylabel="Growth of $1",
        title="Posterior predictive wealth paths",
    )
    axes[0].legend(frameon=False, ncol=2)

    predictive_drawdown = _max_drawdown(paths)
    observed_drawdown = _max_drawdown(observed[None, :])[0]
    axes[1].hist(
        predictive_drawdown,
        bins=40,
        density=True,
        color="#A98BCB",
        alpha=0.8,
    )
    axes[1].axvline(
        observed_drawdown,
        color=OBSERVED,
        linewidth=2,
        label=f"VOO observed: {100 * observed_drawdown:.1f}%",
    )
    axes[1].set(
        xlabel="Maximum drawdown",
        ylabel="Density",
        title="Posterior predictive downside",
    )
    axes[1].legend(frameon=False)
    return fig


def _density(values, bins):
    magnitudes = np.abs(np.asarray(values, dtype="float64").reshape(-1))
    magnitudes = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
    counts = np.histogram(magnitudes, bins=bins)[0]
    return counts / (len(magnitudes) * np.diff(bins))


def _pointwise_hdi(draws, probability):
    """Return the narrowest sample interval in each density bin."""
    interval = max(
        1,
        min(len(draws) - 1, int(np.floor(probability * len(draws)))),
    )
    ordered = np.sort(draws, axis=0)
    widths = ordered[interval:] - ordered[:-interval]
    starts = np.argmin(widths, axis=0)
    columns = np.arange(draws.shape[1])
    return (
        ordered[starts, columns],
        ordered[starts + interval, columns],
    )


def _smooth_densities(densities, bins, sigma):
    widths = np.diff(bins)
    masses = densities * widths
    smoothed = gaussian_filter1d(
        masses, sigma=sigma, axis=1, mode="constant", cval=0
    )
    smoothed *= masses.sum(axis=1, keepdims=True) / smoothed.sum(
        axis=1, keepdims=True
    )
    return smoothed / widths


def plot_gaussian_mmar_comparison(
    observed_returns,
    posterior_paths,
    ticker="VOO",
    n_bins=40,
    hdi_probability=0.92,
):
    """Compare observed return magnitudes with Gaussian and MMAR fits."""
    if isinstance(observed_returns, pd.Series):
        date_label = (
            f"{observed_returns.index.min():%b %d, %Y}–"
            f"{observed_returns.index.max():%b %d, %Y}"
        )
    else:
        date_label = f"{len(observed_returns):,} trading days"

    observed_log = np.log1p(np.asarray(observed_returns).reshape(-1))
    predictive_log = np.log1p(np.asarray(posterior_paths, dtype="float64"))
    observed_magnitude = np.abs(observed_log)
    observed_magnitude = observed_magnitude[observed_magnitude > 0]

    lower = max(float(np.quantile(observed_magnitude, 0.005)), 1e-3)
    upper = 1.05 * float(observed_magnitude.max())
    bins = np.geomspace(lower, upper, n_bins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])

    observed_density = _density(observed_log, bins)
    predictive_densities = np.stack(
        [_density(path, bins) for path in predictive_log]
    )
    predictive_densities = _smooth_densities(
        predictive_densities, bins, sigma=1.5
    )
    predictive_mean = predictive_densities.mean(axis=0)
    predictive_lower, predictive_upper = _pointwise_hdi(
        predictive_densities, hdi_probability
    )

    normal_loc, normal_scale = stats.norm.fit(observed_log)
    x_grid = np.geomspace(lower, upper, 500)
    normal_density = stats.norm.pdf(
        x_grid, normal_loc, normal_scale
    ) + stats.norm.pdf(-x_grid, normal_loc, normal_scale)
    gaussian_99 = stats.foldnorm.ppf(
        0.99, abs(normal_loc) / normal_scale, scale=normal_scale
    )

    occupied = observed_density > 0
    outliers = occupied & (centers > gaussian_99)
    ordinary = occupied & ~outliers
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharex=True, sharey=True)

    model_specs = (
        ("Gaussian fit", x_grid, normal_density, GREEN),
        ("MMAR posterior predictive fit", centers, predictive_mean, PURPLE),
    )
    model_lines = []
    for panel, (ax, (title, x, density, color)) in enumerate(
        zip(axes, model_specs, strict=True)
    ):
        ax.bar(
            bins[:-1][ordinary],
            observed_density[ordinary],
            width=np.diff(bins)[ordinary],
            align="edge",
            color="#C9CDD4",
            edgecolor="#737A86",
            linewidth=0.55,
            alpha=0.82,
        )
        ax.bar(
            bins[:-1][outliers],
            observed_density[outliers],
            width=np.diff(bins)[outliers],
            align="edge",
            color="#D1495B",
            edgecolor="#8E2433",
            linewidth=0.75,
            alpha=0.88,
        )
        if panel == 1:
            ax.fill_between(
                centers,
                np.maximum(predictive_lower, 1e-3),
                predictive_upper,
                where=predictive_upper > 1e-3,
                color=PURPLE,
                alpha=0.24,
                linewidth=0,
            )
        (line,) = ax.plot(
            x,
            np.where(density > 1e-3, density, np.nan),
            color=color,
            linewidth=3,
        )
        model_lines.append(line)
        ax.set(xscale="log", yscale="log", title=title)
        ax.set_xlabel("Absolute daily log-return")
        ax.grid(False, which="minor")

    positive_density = observed_density[observed_density > 0]
    axes[0].set_ylim(
        0.25 * positive_density.min(),
        1.8
        * max(
            observed_density.max(),
            normal_density.max(),
            predictive_upper.max(),
        ),
    )
    axes[0].set_ylabel("Probability density")
    fig.suptitle(
        f"Absolute {ticker} returns: Gaussian vs. MMAR ({date_label})",
        fontsize=19,
    )
    observed_patch = Patch(facecolor="#C9CDD4", edgecolor="#737A86")
    fig.legend(
        [model_lines[0], model_lines[1], observed_patch],
        [
            "Gaussian MLE",
            f"MMAR mean and {100 * hdi_probability:g}% HDI",
            f"{ticker} observed; Gaussian misses red",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig
