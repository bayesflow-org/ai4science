"""Assets and minimal helpers for the univariate MMAR tutorial."""

from pathlib import Path

import pandas as pd

from .model import (
    PARAMETER_LABELS,
    PARAMETER_NAMES,
    PRIOR,
    WINDOW,
    posterior_resimulations,
    prior_table,
    q_scenarios,
    simulate,
    stack_samples,
)
from .plotting import (
    configure_plot_style,
    plot_gaussian_mmar_comparison,
    plot_posterior_predictive,
    plot_prior_predictive,
    plot_q_turbulence,
)

ASSET_DIR = Path(__file__).resolve().parent
MODEL_PATH = ASSET_DIR / "univariate.keras"
CASCADE_GIF_PATH = ASSET_DIR / "fractal_cascade.gif"
MARKET_DATA_PATH = ASSET_DIR / "voo_returns.csv"


def load_market_returns():
    """Load the cached VOO simple returns used in the tutorial."""
    returns = pd.read_csv(MARKET_DATA_PATH, index_col=0, parse_dates=True)
    returns.index.name = "date"
    return returns["VOO"]


__all__ = [
    "CASCADE_GIF_PATH",
    "MARKET_DATA_PATH",
    "MODEL_PATH",
    "PARAMETER_LABELS",
    "PARAMETER_NAMES",
    "PRIOR",
    "WINDOW",
    "configure_plot_style",
    "load_market_returns",
    "plot_gaussian_mmar_comparison",
    "plot_posterior_predictive",
    "plot_prior_predictive",
    "plot_q_turbulence",
    "posterior_resimulations",
    "prior_table",
    "q_scenarios",
    "simulate",
    "stack_samples",
]
