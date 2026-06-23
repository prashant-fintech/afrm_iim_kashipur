#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AFRM Assignment 1 Runner
Models:
1) Historical Simulation
2) Monte Carlo Simulation
3) Variance-Covariance (Delta-Normal) with 250-day rolling window
4) RiskMetrics (EWMA, lambda=0.94)
5) GARCH(1,1) with six distributions
6) GJR-GARCH(1,1) with six distributions
7) EGARCH(1,1) with six distributions

Backtesting summary: % breaches, excess capital, excess loss
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from scipy.stats import norm, t as student_t, skew as scipy_skew, kurtosis as scipy_kurtosis
except Exception:
    norm = None
    student_t = None
    scipy_skew = None
    scipy_kurtosis = None

try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except Exception:
    arch_model = None
    ARCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
ARCH_MODEL_CONFIGS = {
    "garch": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "gjr": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    "egarch": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
}

# The last two distributions (skewged, nig) are approximations built on top of
# the GED and t distributions respectively. Their quantiles are scaled to
# introduce additional skewness/heavy-tail behaviour when arch does not
# expose those families natively.
ARCH_DIST_CONFIGS = [
    {"name": "normal", "arch": "normal"},
    {"name": "t", "arch": "t"},
    {"name": "skewt", "arch": "skewt"},
    {"name": "ged", "arch": "ged"},
    {"name": "skewged", "arch": "ged", "skew_scale": 0.12},
    {"name": "nig", "arch": "t", "kurt_scale": 0.05},
]


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def load_returns(
    path: str, price_col: str = "Close", date_col: str = "Date"
) -> pd.Series:
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    for col in [price_col, "Adj Close", "Price", "ClosePrice", "PX_LAST"]:
        if col in df.columns:
            price_col = col
            break
    if price_col not in df.columns:
        raise ValueError(f"Price column not found. Available: {list(df.columns)}")

    returns = np.log(df[price_col]).diff().dropna()
    returns.name = "r"
    if date_col in df.columns:
        returns.index = df.loc[returns.index, date_col]
    return returns.astype(float)


@dataclass
class BacktestResult:
    breaches_pct: float
    excess_capital: float
    excess_loss: float
    n_obs: int


def compute_excess_metrics(returns: pd.Series, var_series: pd.Series) -> BacktestResult:
    """
    Compare realized losses (-returns) against VaR_t (positive loss threshold).
    Breach when (-r_t) > VaR_t.
    """
    aligned = pd.concat([returns, var_series], axis=1, keys=["r", "VaR"]).dropna()
    if aligned.empty:
        return BacktestResult(np.nan, np.nan, np.nan, 0)

    losses = -aligned["r"]
    var_threshold = aligned["VaR"].abs()
    breaches = losses > var_threshold

    excess_capital = (losses - var_threshold).clip(lower=0).sum()
    excess_loss = losses[breaches].sum()
    return BacktestResult(
        breaches_pct=float(breaches.mean() * 100.0),
        excess_capital=float(excess_capital),
        excess_loss=float(excess_loss),
        n_obs=len(aligned),
    )


# ---------------------------------------------------------------------------
# Non-ARCH VaR models
# ---------------------------------------------------------------------------
def hist_var(returns: pd.Series, alpha: float, window: int) -> pd.Series:
    def loss_quantile(x: pd.Series) -> float:
        return float(np.quantile(-x, alpha))

    return returns.rolling(window).apply(loss_quantile, raw=False).rename("VaR_hist")


def varcov_var(returns: pd.Series, alpha: float, window: int) -> pd.Series:
    z = abs(norm.ppf(1 - alpha)) if norm is not None else 2.326347874
    sigma = returns.rolling(window).std()
    return (z * sigma).rename("VaR_varcov")


def riskmetrics_var(returns: pd.Series, alpha: float, lam: float = 0.94) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, name="VaR_riskmetrics")
    z = abs(norm.ppf(1 - alpha)) if norm is not None else 2.326347874
    ewma = pd.Series(index=returns.index, dtype=float)
    ewma.iloc[0] = returns.var()
    for t in range(1, len(returns)):
        ewma.iloc[t] = lam * ewma.iloc[t - 1] + (1 - lam) * returns.iloc[t - 1] ** 2
    sigma = np.sqrt(ewma)
    return (z * sigma).rename("VaR_riskmetrics")


def mc_var_normal_t(
    returns: pd.Series,
    alpha: float,
    window: int,
    sims: int = 10000,
    dist: str = "normal",
) -> pd.Series:
    rng = np.random.default_rng(42 if dist == "normal" else 99)
    mu_roll = returns.rolling(window).mean()
    sigma_roll = returns.rolling(window).std()
    df = 6
    out = []
    for idx in range(len(returns)):
        if idx < window:
            out.append(np.nan)
            continue
        mu = mu_roll.iloc[idx]
        sigma = sigma_roll.iloc[idx]
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            out.append(np.nan)
            continue
        if dist == "normal":
            sims_r = rng.normal(mu, sigma, sims)
        elif dist == "t":
            if student_t is not None:
                sims_r = mu + sigma * student_t.rvs(df, size=sims, random_state=rng)
            else:
                sims_r = mu + sigma * rng.standard_t(df, size=sims)
        else:
            raise ValueError("dist must be 'normal' or 't'")
        losses = -sims_r
        out.append(np.quantile(losses, alpha))
    return pd.Series(out, index=returns.index, name=f"VaR_mc_{dist}")


# ---------------------------------------------------------------------------
# ARCH-based VaR models
# ---------------------------------------------------------------------------
def _quantile_from_fit(res, alpha: float, cfg: Dict[str, float]) -> float:
    level = max(min(1 - alpha, 1 - 1e-6), 1e-6)
    try:
        q = float(res.distribution.ppf(level))
    except Exception:
        if student_t is not None:
            q = float(student_t.ppf(level, 6))
        elif norm is not None:
            q = float(norm.ppf(level))
        else:
            q = -2.326347874
    q = abs(q)
    resid = res.std_resid.dropna()
    if cfg.get("skew_scale") and len(resid) > 10:
        if scipy_skew is not None:
            skewness = abs(float(scipy_skew(resid, bias=False)))
        else:
            skewness = abs(float(resid.skew()))
        q *= 1 + cfg["skew_scale"] * skewness
    if cfg.get("kurt_scale") and len(resid) > 10:
        if scipy_kurtosis is not None:
            kurt_value = float(scipy_kurtosis(resid, fisher=True, bias=False))
        else:
            kurt_value = float(resid.kurt())
        kurt = max(kurt_value, 0.0)
        q *= 1 + cfg["kurt_scale"] * kurt
    return q


def fit_forecast_var_arch(
    returns: pd.Series,
    alpha: float,
    window: int,
    model_type: str = "garch",
) -> Dict[str, pd.Series]:
    if not ARCH_AVAILABLE:
        raise RuntimeError("arch package is not available. Install 'arch' first.")
    if model_type not in ARCH_MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type '{model_type}'")

    r = returns.dropna()
    if len(r) < max(window, 50) + 5:
        raise ValueError("Not enough observations for ARCH fitting.")

    start_idx = min(len(r) - 1, max(window, 1))
    model_cfg = ARCH_MODEL_CONFIGS[model_type]

    base_results: Dict[str, Optional[Dict[str, pd.Series]]] = {}
    unique_arch_dists = {cfg["arch"] for cfg in ARCH_DIST_CONFIGS}
    for base_dist in sorted(unique_arch_dists):
        am = arch_model(
            r,
            mean="Constant",
            vol=model_cfg["vol"],
            p=model_cfg["p"],
            o=model_cfg["o"],
            q=model_cfg["q"],
            dist=base_dist,
            rescale=False,
        )
        try:
            res = am.fit(disp="off")
        except Exception as exc:
            warnings.warn(f"{model_type.upper()}-{base_dist} fit failed: {exc}")
            base_results[base_dist] = None
            continue

        fcast = res.forecast(horizon=1, start=start_idx, reindex=True)
        sigma = np.sqrt(fcast.variance["h.1"])
        base_results[base_dist] = {"result": res, "sigma": sigma}

    out: Dict[str, pd.Series] = {}
    for cfg in ARCH_DIST_CONFIGS:
        name = f"VaR_{model_type}_{cfg['name']}"
        base = base_results.get(cfg["arch"])
        if not base:
            out[name] = pd.Series(np.nan, index=r.index, name=name)
            continue
        res = base["result"]
        sigma = base["sigma"]
        q = _quantile_from_fit(res, alpha, cfg)
        out[name] = (q * sigma).rename(name)
    return out


# ---------------------------------------------------------------------------
# Pipeline + CLI
# ---------------------------------------------------------------------------
def run_pipeline(
    csv_path: str,
    alpha: float = 0.99,
    window: int = 250,
    price_col: str = "Close",
    date_col: str = "Date",
) -> pd.DataFrame:
    returns = load_returns(csv_path, price_col=price_col, date_col=date_col)
    outputs: Dict[str, pd.Series] = {}

    outputs["VaR_hist"] = hist_var(returns, alpha, window)
    outputs["VaR_mc_normal"] = mc_var_normal_t(returns, alpha, window, dist="normal")
    outputs["VaR_mc_t"] = mc_var_normal_t(returns, alpha, window, dist="t")
    outputs["VaR_varcov"] = varcov_var(returns, alpha, window)
    outputs["VaR_riskmetrics"] = riskmetrics_var(returns, alpha)

    if ARCH_AVAILABLE:
        for model_type in ARCH_MODEL_CONFIGS.keys():
            var_dict = fit_forecast_var_arch(returns, alpha, window, model_type=model_type)
            outputs.update(var_dict)
    else:
        warnings.warn("arch not available. Skipping GARCH-based models.")

    var_df = pd.concat(outputs.values(), axis=1)

    records = []
    for name, series in var_df.items():
        bt = compute_excess_metrics(returns, series)
        records.append(
            {
                "Model": name,
                "% Breaches": bt.breaches_pct,
                "Excess Capital": bt.excess_capital,
                "Excess Loss": bt.excess_loss,
                "Obs": bt.n_obs,
            }
        )
    summary = pd.DataFrame(records).sort_values("% Breaches")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AFRM Assignment 1 Runner")
    parser.add_argument(
        "--file", type=str, required=True, help="Path to CSV with Date and Close columns"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.99, help="VaR confidence level, e.g., 0.99"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=250,
        help="Rolling window size for historical / MC / variance-covariance models",
    )
    parser.add_argument("--price_col", type=str, default="Close")
    parser.add_argument("--date_col", type=str, default="Date")
    args = parser.parse_args()

    summary = run_pipeline(
        args.file,
        alpha=args.alpha,
        window=args.window,
        price_col=args.price_col,
        date_col=args.date_col,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
