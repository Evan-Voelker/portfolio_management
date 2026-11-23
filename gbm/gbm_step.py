"""Daily Monte Carlo step for a rebalanced, regime-switching portfolio.

Each path is a correlated geometric Brownian motion. Drift and volatility
come from a three-state regime unless the caller passes one drift for the
whole run. The valuation, the signal, and the regime each keep their own
memory, and they do not share a calendar:

- The price step is one day of length `dt` (1/252 of a year in `main`).
- The regime draws on its own interval, 21 trading days unless the sheet
  says otherwise. A draw at the end of a day applies on the next day.
- A momentum sleeve rotates every 25 trading days. The day's growth is
  credited to the holding that was already in place.
- Allocation bands are checked every 21 trading days, or every day when
  dynamic rebalance is on. One trigger asset outside its band snaps the
  whole path back to target weights.

Signals and the trailing-return add-on compound market growth, including
assets the book does not hold. A rebalance moves dollars. It does not
rewrite those shadow prices. The warmup that fills an SMA before the open
does not age the regime, so the first regime draws use the untilted matrix
and every path opens in Churn.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from gbm.asset_simulation_matrix import CORRELATION_TICKERS, load_asset_cholesky
from gbm.bitcoin import MATURE_BTC_STRESS_BLEND, MATURE_BTC_STRESS_CORRELATION, restress_bitcoin
from gbm.dbmf import (
    CHURN_EQUITY_CORRELATION,
    EXPANSION_EQUITY_CORRELATION,
    LIQUIDATION_EQUITY_CORRELATION,
    is_dbmf,
    restress_managed_futures,
)
from gbm.expenses import apply_expense_ratios, load_expense_ratios, overlay_model_expenses
from gbm.regime import (
    CASH_FLIGHT_TICKERS,
    CASH_TICKER,
    CHURN_VOL_GRACE_MONTHS,
    CHURN_VOL_MONTHLY_STEP,
    SMA_PROBABILITY_PER_STD,
    SMA_TREND_DAYS,
    SMA_Z_CAP,
    TRADING_DAYS_PER_MONTH,
    RegimeBook,
    _row_for_label,
    align_regime_parameters,
    apply_cash_flight,
    load_transition_matrix,
    read_labeled_cell,
    transition_sheet_rows,
    trigger_index,
)
from gbm.momentum import MOMENTUM_LOOKBACK_DAYS, MomentumBook, load_momentum_drift, overlay_sheet_momentum
from gbm.signals import MOMENTUM_PICKS, SignalBook, cash_asset_index, resolve_asset_signals

# Lower-triangular Cholesky factor L. Correlation is L @ L.T.
# Rebuilt from prices only when gbm/asset_cholesky.csv is missing or the wrong size.
ASSET_CORRELATION = load_asset_cholesky().to_numpy()


def correlation_for_labels(labels, tickers=None, factor=None):
    """Reorder a Cholesky factor's correlation into the sheet's ticker order.

    The stored object is a lower-triangular factor, so the correlation is
    rebuilt as L @ L.T and then a square submatrix is copied into label
    order. A label with no row stays on the identity: uncorrelated with
    every other name, and with any other unknown name. `tickers` and
    `factor` select the matrix built for another sheet.
    """
    universe = list(CORRELATION_TICKERS if tickers is None else tickers)
    matrix_factor = ASSET_CORRELATION if factor is None else np.asarray(factor, dtype=float)
    keys = []
    for label in labels:
        try:
            keys.append(_correlation_key(label, universe))
        except KeyError:
            keys.append(None)
    matrix = np.eye(len(labels))
    known = [position for position, key in enumerate(keys) if key is not None]
    if known:
        source = [universe.index(keys[position]) for position in known]
        correlation = matrix_factor @ matrix_factor.T
        matrix[np.ix_(known, known)] = correlation[np.ix_(source, source)]
    return matrix


def _correlation_key(label, universe=None):
    """Map a sleeve label onto a row of the correlation universe.

    An exact ticker wins. Otherwise the first slash-separated piece that
    exists in the universe wins, so VOO/VIIX uses the VOO row. BTC and IBIT
    share the BTC row. Anything else is missing and stays independent.
    """
    universe = CORRELATION_TICKERS if universe is None else universe
    text = str(label).strip()
    if text in universe:
        return text
    for part in text.split("/"):
        part = part.strip()
        if part in universe:
            return part
    if text in ("BTC", "IBIT") and "BTC" in universe:
        return "BTC"
    raise KeyError(f"{label} has no row in the correlation matrix")


def simulation_cholesky(labels, sheet_name="Asset Parameters"):
    """Cholesky factor for the model on `sheet_name`.

    Asset Parameters keeps the saved multi-asset factor. Any other sheet loads,
    or builds, the factor whose rows are that sheet's tickers.
    """
    if sheet_name in (None, "", "Asset Parameters"):
        return verify_and_generate_cholesky(correlation_for_labels(labels))
    factor = load_asset_cholesky(sheet_name=sheet_name)
    return verify_and_generate_cholesky(
        correlation_for_labels(labels, tickers=list(factor.index), factor=factor.to_numpy())
    )


def verify_and_generate_cholesky(matrix):
    """Lower-triangular L such that L @ L.T recovers the correlation matrix.

    A matrix that is already positive definite is factored directly. One
    that is not, usually after a hand blend, gets 1e-8 added to the
    diagonal and is factored again. Regime matrices go through
    `nearest_correlation` first, so this ridge is the backstop.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        epsilon = 1e-8
        smoothed_matrix = matrix + np.diag(np.ones(matrix.shape[0]) * epsilon)
        return np.linalg.cholesky(smoothed_matrix)


# Liquidation blends this much of a flat correlation into the historical matrix.
LIQUIDATION_BLEND = 0.7
LIQUIDATION_CORRELATION = 0.85
# Churn keeps this share of every historical off-diagonal.
CHURN_CORRELATION_SCALE = 0.75

CHURN_SCALE_LABEL = "Churn Correlation Scale"
LIQUIDATION_BLEND_LABEL = "Liquidation Blend"
LIQUIDATION_CORRELATION_LABEL = "Liquidation Correlation"
BITCOIN_STRESS_CORRELATION_LABEL = "Bitcoin Stress Correlation"
BITCOIN_STRESS_BLEND_LABEL = "Bitcoin Stress Blend"
DBMF_EXPANSION_LABEL = "DBMF Expansion Equity Correlation"
DBMF_CHURN_LABEL = "DBMF Churn Equity Correlation"
DBMF_LIQUIDATION_LABEL = "DBMF Liquidation Equity Correlation"
CHURN_GRACE_LABEL = "Churn Volatility Grace Months"
CHURN_STEP_LABEL = "Churn Volatility Monthly Step"
CASH_FLIGHT_LABEL = "Cash Flight Tickers"


@dataclass(frozen=True)
class SimulationAssumptions:
    """Stress, churn-vol, and cash-flight settings. The sheet replaces any field it names."""

    churn_correlation_scale: float = CHURN_CORRELATION_SCALE
    liquidation_blend: float = LIQUIDATION_BLEND
    liquidation_correlation: float = LIQUIDATION_CORRELATION
    bitcoin_stress_correlation: float = MATURE_BTC_STRESS_CORRELATION
    bitcoin_stress_blend: float = MATURE_BTC_STRESS_BLEND
    dbmf_expansion_equity: float = EXPANSION_EQUITY_CORRELATION
    dbmf_churn_equity: float = CHURN_EQUITY_CORRELATION
    dbmf_liquidation_equity: float = LIQUIDATION_EQUITY_CORRELATION
    churn_vol_grace_months: int = CHURN_VOL_GRACE_MONTHS
    churn_vol_monthly_step: float = CHURN_VOL_MONTHLY_STEP
    cash_flight: tuple = CASH_FLIGHT_TICKERS


def load_simulation_assumptions(path, sheet="Transition Matrix"):
    """Read the assumption block. A missing label keeps the code default.

    A present label, including a blank cash-flight cell, is the value the
    run uses. Correlation entries may be negative. Scales and blends stay
    inside the unit interval.
    """
    rows = transition_sheet_rows(path, sheet)
    updates = {}

    def number(label, key, check):
        found, raw = read_labeled_cell(rows, label)
        if not found or raw is None or (isinstance(raw, str) and not str(raw).strip()):
            return
        value = float(raw)
        if not np.isfinite(value) or not check(value):
            raise ValueError(f"{label} is outside the range the simulation can use")
        updates[key] = value

    unit = lambda value: 0.0 <= value <= 1.0
    correlation = lambda value: -1.0 <= value <= 1.0
    number(CHURN_SCALE_LABEL, "churn_correlation_scale", lambda value: 0.0 < value <= 1.0)
    number(LIQUIDATION_BLEND_LABEL, "liquidation_blend", unit)
    number(LIQUIDATION_CORRELATION_LABEL, "liquidation_correlation", correlation)
    number(BITCOIN_STRESS_CORRELATION_LABEL, "bitcoin_stress_correlation", correlation)
    number(BITCOIN_STRESS_BLEND_LABEL, "bitcoin_stress_blend", unit)
    number(DBMF_EXPANSION_LABEL, "dbmf_expansion_equity", correlation)
    number(DBMF_CHURN_LABEL, "dbmf_churn_equity", correlation)
    number(DBMF_LIQUIDATION_LABEL, "dbmf_liquidation_equity", correlation)
    number(CHURN_STEP_LABEL, "churn_vol_monthly_step", lambda value: value >= 0.0)
    found_grace, raw_grace = read_labeled_cell(rows, CHURN_GRACE_LABEL)
    if found_grace and raw_grace is not None and not (isinstance(raw_grace, str) and not str(raw_grace).strip()):
        grace = float(raw_grace)
        if not grace.is_integer() or grace < 1:
            raise ValueError(f"{CHURN_GRACE_LABEL} must be a positive whole number of months")
        updates["churn_vol_grace_months"] = int(grace)
    found_flight, raw_flight = read_labeled_cell(rows, CASH_FLIGHT_LABEL)
    if found_flight:
        text = "" if raw_flight is None else str(raw_flight).strip()
        if text.lower() in ("", "none"):
            updates["cash_flight"] = ()
        else:
            updates["cash_flight"] = tuple(part.strip() for part in text.replace(";", ",").split(",") if part.strip())
    return replace(SimulationAssumptions(), **updates)


def nearest_correlation(matrix, threshold=1e-15, n_fact=100):
    """Nearest correlation matrix with eigenvalues at least `threshold`.

    Alternates two projections. One clips eigenvalues of the running
    residual so the matrix is positive semidefinite. The other writes the
    diagonal back to 1. The residual is carried forward so the next
    eigenvalue clip is applied to the difference, which is the Higham
    alternating projection a hand-built blend needs before a Cholesky
    factor exists. A matrix that is already past the threshold is returned
    as-is. The loop does not promise to keep a prescribed off-diagonal:
    the 0.85 liquidation blend and a DBMF equity override can both move.
    """
    updated = np.array(matrix, dtype=float, copy=True)
    if updated.ndim != 2 or updated.shape[0] != updated.shape[1]:
        raise ValueError("correlation matrix must be square")
    updated = (updated + updated.T) / 2.0
    if np.linalg.eigvalsh(updated).min() >= threshold:
        return updated

    difference = np.zeros(updated.shape)
    diagonal = np.diag_indices(updated.shape[0])
    for _ in range(max(int(updated.shape[0] * n_fact), 1)):
        adjusted = updated - difference
        eigenvalues, eigenvectors = np.linalg.eigh(adjusted)
        below = eigenvalues < threshold
        if not np.any(below):
            projected = adjusted
        else:
            eigenvalues = eigenvalues.copy()
            eigenvalues[below] = threshold
            projected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        difference = projected - adjusted
        updated = projected.copy()
        updated[diagonal] = 1.0
        updated = (updated + updated.T) / 2.0
        if np.linalg.norm(projected - adjusted) <= threshold:
            break
    if np.linalg.eigvalsh(updated).min() < threshold:
        eigenvalues, eigenvectors = np.linalg.eigh(updated)
        eigenvalues = np.maximum(eigenvalues, threshold)
        repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        scale = np.sqrt(np.diag(repaired))
        updated = repaired / np.outer(scale, scale)
    return updated


def regime_correlation_matrices(baseline, labels=None, assumptions=None):
    """Expansion, Churn, and Liquidation correlations from one historical matrix.

    Expansion keeps the historical matrix. Churn multiplies off-diagonals by
    the churn scale and restores the unit diagonal. Liquidation blends the
    historical matrix with a flat correlation, then restores the unit diagonal.
    The scale, the blend, and the flat level come from `assumptions` when the
    transition sheet named them, and otherwise from the published defaults.
    When `labels` names BTC, that row joins the risk complex partway toward
    the bitcoin stress correlation and keeps its historical correlation with
    cash. When `labels` names DBMF, that row uses the sheet's equity
    correlation instead of the flat blend. Churn and Liquidation are projected
    onto a positive-definite correlation so the factor exists. Expansion is
    projected too once DBMF has rewritten a row.
    """
    assumptions = SimulationAssumptions() if assumptions is None else assumptions
    baseline = np.asarray(baseline, dtype=float)
    baseline = (baseline + baseline.T) / 2.0
    np.fill_diagonal(baseline, 1.0)
    churn = baseline * assumptions.churn_correlation_scale
    np.fill_diagonal(churn, 1.0)
    width = baseline.shape[0]
    liquidation = (1.0 - assumptions.liquidation_blend) * baseline + assumptions.liquidation_blend * np.full(
        (width, width), assumptions.liquidation_correlation
    )
    np.fill_diagonal(liquidation, 1.0)
    expansion = baseline
    if labels is not None:
        liquidation = restress_bitcoin(
            baseline,
            liquidation,
            labels,
            stress_correlation=assumptions.bitcoin_stress_correlation,
            stress_blend=assumptions.bitcoin_stress_blend,
        )
        expansion, churn, liquidation = restress_managed_futures(
            baseline,
            expansion,
            churn,
            liquidation,
            labels,
            equity=(
                assumptions.dbmf_expansion_equity,
                assumptions.dbmf_churn_equity,
                assumptions.dbmf_liquidation_equity,
            ),
        )
    if labels is not None and any(is_dbmf(label) for label in labels):
        expansion = nearest_correlation(expansion)
    return expansion, nearest_correlation(churn), nearest_correlation(liquidation)


def regime_cholesky_factors(baseline, labels=None, assumptions=None):
    """Lower-triangular factors for Expansion, Churn, and Liquidation, in that order.

    Stack axis 0 is the regime number minus one. `apply_cholesky` indexes it
    with states numbered from 1, so state 1 uses the historical factor.
    `assumptions` is the transition-sheet block. Missing labels keep the
    published stress numbers.
    """
    return np.stack([
        verify_and_generate_cholesky(matrix)
        for matrix in regime_correlation_matrices(baseline, labels, assumptions)
    ])


def apply_cholesky(shocks, cholesky, states=None):
    """Map independent shocks through one factor, one factor per path, or one factor per regime.

    A single factor has shape (assets, assets). A per-path stack has shape
    (paths, assets, assets). A regime stack has shape (states, assets, assets)
    and `states` picks a row for each path, numbered from 1. Every factor is
    lower triangular, and the shocks are multiplied by L.T.
    """
    if cholesky is None:
        return shocks
    factor = np.asarray(cholesky, dtype=float)
    if factor.ndim == 2:
        return shocks @ factor.T
    if states is None:
        return np.einsum("pi,pji->pj", shocks, factor)
    correlated = np.empty(shocks.shape, dtype=float)
    states = np.asarray(states)
    for index in range(factor.shape[0]):
        mask = states == index + 1
        if not np.any(mask):
            continue
        correlated[mask] = shocks[mask] @ factor[index].T
    return correlated


def gbm_growth(drift, volatility, dt, cholesky=None, states=None):
    """One correlated growth factor per path and asset.

    The step is the exact lognormal discretization of dS/S = mu dt + sigma dW:
    exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z). The -0.5 sigma^2 term is
    the Ito correction, so the median grows slower than the drift and a
    high-vol name gives up about half its variance per year. Z is standard
    normal, then correlated. A zero holding still gets a factor, because
    the signal and the momentum add-on compound this shadow price.
    """
    drift = np.asarray(drift, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    num_paths, num_assets = drift.shape
    shocks = np.random.standard_normal((num_paths, num_assets))
    shocks = apply_cholesky(shocks, cholesky, states)
    return np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks)


def total_gbm(drift, volatility, dt, num_days, num_paths=None, cholesky=None, states=None, momentum=None):
    """Correlated shadow prices, oldest first, for the days before the book opens.

    Every asset starts at 1. The last row is the close on the opening day.
    `num_days` is the longest SMA Enter or SMA Exit window, including the closes
    SMA-exclusion leaves out of the average. The sleeve can then open in cash or
    in a momentum bucket. This path is not the portfolio valuation. `momentum`
    adds the centered trailing-return drift and keeps the shadow price.
    """
    drift = np.asarray(drift, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    if drift.ndim == 1:
        if num_paths is None:
            raise ValueError("num_paths is required when drift is one row")
        drift = np.broadcast_to(drift, (int(num_paths), drift.shape[0]))
    elif num_paths is not None and drift.shape[0] != int(num_paths):
        raise ValueError("drift rows must match num_paths")
    num_paths, num_assets = drift.shape
    volatility = np.broadcast_to(volatility, drift.shape)
    num_days = int(num_days)
    if num_days < 0:
        raise ValueError("num_days must be non-negative")
    closes = np.empty((num_days, num_paths, num_assets))
    level = np.ones((num_paths, num_assets))
    for day in range(num_days):
        step_drift = drift if momentum is None else momentum.adjust(drift, dt)
        growth = gbm_growth(step_drift, volatility, dt, cholesky, states)
        if momentum is not None:
            momentum.observe(growth)
        level = level * growth
        closes[day] = level
    return closes


def open_from_total_gbm(signal, drift, volatility, dt, target_allocation, cholesky=None, states=None, momentum=None):
    """Fill the SMA from a shadow GBM, then open each path in cash or a bucket.

    Scoring starts from cash, so the hurdle is SMA-Enter. With beat_cash on, a share
    that does not beat the cash rate stays in SGOV. With it off, a share stays in SGOV
    when its price is not above that average.
    """
    closes = total_gbm(
        drift,
        volatility,
        dt,
        signal.lead_days,
        signal.selection.shape[0],
        cholesky,
        states,
        momentum,
    )
    signal.load_history(closes)
    signal.start(in_cash=True)
    signal.choose_momentum(target_allocation)


def gbm_step_vectorized_3d(current_prices, drift, volatility, dt, cholesky=None, states=None):
    """
    Simulates a step forward for all paths and assets simultaneously.
    Shapes:
      current_prices, drift, volatility: (num_paths, num_assets)
    cholesky is the lower-triangular factor of the asset correlation matrix,
    or one such factor per path. Independent shocks Z become Z @ L.T, so their
    covariance is L @ L.T.
    """
    return np.asarray(current_prices, dtype=float) * gbm_growth(drift, volatility, dt, cholesky, states)


def growth_factors(drift, volatility, held_drift, held_volatility, dt, cholesky=None, states=None):
    """One shared shock, two drifts.

    Z is drawn once and correlated once. The market factor uses the asset's
    own drift and volatility, and that is what the SMA observes. The held
    factor uses the cash drift and cash volatility on sleeves the signal has
    turned off, and that is what the book earns. The momentum-sleeve path
    does not use this split: it moves the dollars instead of swapping the
    drift.
    """
    num_paths, num_assets = drift.shape
    Z = apply_cholesky(np.random.standard_normal((num_paths, num_assets)), cholesky, states)
    scale = np.sqrt(dt) * Z

    def factor(step_drift, step_volatility):
        return np.exp((step_drift - 0.5 * step_volatility**2) * dt + step_volatility * scale)

    return factor(drift, volatility), factor(held_drift, held_volatility)


def dynamic_rebalance(valuations, lower_threshold, upper_threshold, target_weights, trigger_mask=None):
    """Snap every path that crossed a band back to its target weights.

    valuations is (num_paths, num_assets). Weights are that row divided by
    its sum. A breach is a weight at or outside its band. Any one breach on
    a path replaces the whole row with portfolio value times target weights.
    Names that stayed inside their bands are reset too. A path with no
    breach keeps the drifted valuation, so the trade is all or nothing.

    trigger_mask limits which assets can cause that snap. A one-dimensional
    mask applies to every path. A (num_paths, num_assets) mask can differ by
    path, which is how a momentum book ignores sleeve names and zero-target
    rows. Breaches outside the mask are cleared before they are counted, so
    the returned breach grids match the assets that were allowed to trade.
    """
    valuations = np.asarray(valuations, dtype=float)
    portfolio_value = valuations.sum(axis=1, keepdims=True)
    weights = valuations / portfolio_value
    lower_breach = weights <= np.asarray(lower_threshold, dtype=float)
    upper_breach = weights >= np.asarray(upper_threshold, dtype=float)
    breach = lower_breach | upper_breach
    if trigger_mask is None:
        path_trigger = breach.any(axis=1, keepdims=True)
    else:
        mask = np.asarray(trigger_mask, dtype=bool)
        if mask.ndim == 1:
            path_trigger = breach[:, mask].any(axis=1, keepdims=True)
            lower_breach = lower_breach.copy()
            upper_breach = upper_breach.copy()
            lower_breach[:, ~mask] = False
            upper_breach[:, ~mask] = False
        else:
            lower_breach = lower_breach & mask
            upper_breach = upper_breach & mask
            path_trigger = (lower_breach | upper_breach).any(axis=1, keepdims=True)
    rebalanced = portfolio_value * np.asarray(target_weights, dtype=float)
    return np.where(path_trigger, rebalanced, valuations), lower_breach, upper_breach


def rotate_momentum_sleeve(valuations, selection, momentum_mask, weights=None):
    """Move each path's momentum dollars onto the selected asset or bucket.

    The sleeve is the sum of columns in `momentum_mask`. Those columns are
    zeroed, and the sum is written back. Alternates are outside the mask, so
    they stay at the value they just grew into. Without `weights`, the whole
    sum lands on `selection`. With `weights`, each asset receives that
    fraction of the sum, and each row of `weights` sums to 1.

    Parked dollars are visible to this sum only when they sit on a masked
    column. The cash row is masked when its asset type is Momentum, which is
    how a sleeve that was swept onto SGOV can be picked up on the way out.
    """
    valuations = np.array(valuations, dtype=float, copy=True)
    mask = np.asarray(momentum_mask, dtype=bool)
    sleeve = valuations[:, mask].sum(axis=1)
    valuations[:, mask] = 0.0
    if weights is None:
        valuations[np.arange(valuations.shape[0]), selection] = sleeve
    else:
        valuations += sleeve[:, None] * np.asarray(weights, dtype=float)
    return valuations


def settle_momentum_book(valuations, signal, targets=None):
    """Rotate the momentum sleeve, and park or restore gated dollars with the cash decision.

    Order matters. A path that just left SGOV splits that cash pile before the
    rotate, so gated assets are funded from the pile and the remainder is what
    the sleeve sum still contains. A path that is in SGOV rotates first, which
    parks the sleeve on SGOV, then sweeps gated dollars onto that same pile.
    A path that stayed invested only rotates. Alternates are never part of
    either move.
    """
    valuations = np.array(valuations, dtype=float, copy=True)
    exited = signal.was_in_cash & ~signal.in_cash
    if signal.gated_mask.any() and np.any(exited):
        if targets is None:
            raise ValueError("gated assets need target weights when the sleeve leaves cash")
        valuations = _restore_gated(valuations, signal, np.asarray(targets, dtype=float).reshape(-1), exited)
    if targets is None and signal.has_buckets:
        raise ValueError("bucketed momentum needs target weights")
    shares = signal.sleeve_weights(targets)
    valuations = rotate_momentum_sleeve(valuations, signal.selection, signal.momentum_mask, shares)
    if signal.gated_mask.any() and np.any(signal.in_cash):
        valuations = _sweep_gated(valuations, signal, signal.in_cash)
    return valuations


def _restore_gated(valuations, signal, targets, paths):
    """Give gated assets their share of the SGOV pile on the trade that leaves cash.

    The pile is treated as the momentum sleeve plus the gated targets. The
    sleeve weight is one minus the alternate targets, and gated names are
    alternates, so the denominator is that sleeve plus the gated targets.
    Each gated asset takes cash times its target over that denominator. The
    rest stays on SGOV for the rotate that follows. Paths that did not leave
    cash are not touched.
    """
    if signal.cash_index is None:
        raise ValueError("gated sleeve needs SGOV")
    sleeve = 1.0 - float(targets[signal.alternate_mask].sum())
    gated_targets = targets[signal.gated_mask]
    denom = sleeve + float(gated_targets.sum())
    if denom <= 0:
        return valuations
    path_index = np.flatnonzero(paths)
    cash = valuations[path_index, signal.cash_index]
    taken = cash[:, None] * (gated_targets / denom)
    gated_index = np.flatnonzero(signal.gated_mask)
    valuations[np.ix_(path_index, gated_index)] += taken
    valuations[path_index, signal.cash_index] = cash - taken.sum(axis=1)
    return valuations


def _sweep_gated(valuations, signal, paths):
    """Move gated dollars onto SGOV for paths that are parked in cash.

    The gated columns are zeroed and their sum is added to the cash column.
    Other alternates are left where they are. Runs after the rotate, so the
    sleeve is already sitting on SGOV and the gated dollars join it.
    """
    if signal.cash_index is None:
        raise ValueError("gated sleeve needs SGOV")
    path_index = np.flatnonzero(paths)
    gated_index = np.flatnonzero(signal.gated_mask)
    moved = valuations[np.ix_(path_index, gated_index)].sum(axis=1)
    valuations[np.ix_(path_index, gated_index)] = 0.0
    valuations[path_index, signal.cash_index] += moved
    return valuations


def momentum_shift(valuations, signal, market_growth, target_allocation=None):
    """Mark the book to market and rotate the momentum sleeve on its trade day.

    The book is multiplied by `market_growth` first, then the signal records
    that same factor, and only then does the trade clock advance. The new
    close is inside the SMA before the ranking. This function does not
    consult allocation bands. Alternates keep the value they grew into.
    Gated assets sit in SGOV while the path is in cash, and are split back
    out of that cash pile on the trade that leaves. The signal's trade
    interval decides the day: 25 trading days by default. Returns the new
    valuations and whether this step rotated the sleeve.
    """
    if not signal.uses_momentum_sleeve:
        raise ValueError("momentum_shift requires assets labeled Momentum")
    grown = np.asarray(valuations, dtype=float) * np.asarray(market_growth, dtype=float)
    signal.observe(np.asarray(market_growth, dtype=float))
    if not signal.consider_trade(target_allocation):
        return grown, False
    return settle_momentum_book(grown, signal, target_allocation), True


def monthly_momentum(prices, drift, volatility, dt, signal, target_allocation, cholesky=None, states=None, with_targets=True):
    """One day for a book that has no momentum sleeve.

    `signal.apply` copies cash drift and cash volatility onto every sleeve
    `held()` says is out. Both factors share one shock, so the SMA and the
    valuation see the same random draw at different parameters. The returned
    targets have shape (num_paths, num_assets) when a path can select a
    different tactical group. They are built only on band-check days. The
    caller observes `market_growth` after the book is marked, which is the
    opposite order from `momentum_shift`.
    """
    held_drift, held_volatility = signal.apply(drift, volatility)
    market_growth, held_growth = growth_factors(
        drift, volatility, held_drift, held_volatility, dt, cholesky, states
    )
    targets = None
    if with_targets:
        targets = np.asarray(signal.target_weights(target_allocation), dtype=float)
    return prices * held_growth, targets, market_growth


def simulate_with_rebalance(S0, drift, volatility, T, dt, num_paths, lower_threshold, upper_threshold, target_allocation, initial_value=None, cholesky=None, regime_drift=None, regime_volatility=None, trigger_index=None, initial_states=None, regime_states=None, signal=None, rebalance_interval=TRADING_DAYS_PER_MONTH, momentum_days=None, momentum_log=None, transition=None, transition_interval=None, regime_rng=None, lambda_mom=None, momentum_center=None, momentum_lookback=MOMENTUM_LOOKBACK_DAYS, sma_window=None, sma_per_std=SMA_PROBABILITY_PER_STD, sma_z_cap=SMA_Z_CAP, labels=None, assumptions=None):
    """Advance every path for `T` years and rebalance on schedule.

    One step is `dt` years. `num_steps` is `int(T / dt)`, so a 10-year run
    at 1/252 is 2,520 priced days. The opening row is not a step. Four
    clocks run inside the loop and none of them is derived from the others:
    the price step, the regime interval, the momentum trade interval, and
    the band-check interval.

    A momentum sleeve grows every name at its market drift, then concentrates
    the masked dollars on the selected names. The shadow price keeps moving
    for a name whose valuation is zero. Alternates can be the only names
    that trip a band. On a day that is both a trade and a breach, the breach
    wins and the target book is the post-trade book.

    A book with no momentum sleeve uses `monthly_momentum`. Unheld sleeves
    earn the cash parameters. An all-alternate book skips that swap and holds
    the spreadsheet targets. A blank signal is a static book: one growth
    factor, and every asset can trip a band.

    Before the first priced day, a momentum sleeve with an empty history is
    filled with `lead_days` of shadow prices under the opening regime. That
    warmup consumes random draws, ages the momentum-drift book, and sets the
    opening weights. It does not call `regime.observe`. Price shocks and,
    when `regime_rng` is omitted, regime draws share NumPy's global
    generator, in that order each day.

    `regime_states[t]` is the state that produced the return from t-1 to t.
    Row 0 repeats the opening state. Band counts record breaches on assets
    the mask allowed to trade.
    """
    interval = int(rebalance_interval)
    if interval < 1:
        raise ValueError("rebalance_interval must be at least 1 trading day")
    num_steps = int(T / dt)
    num_assets = len(S0)
    time_axis = np.linspace(0, T, num_steps + 1)
    
    # Axis 0 is the open plus each priced day. Axis 1 is a path. Axis 2 is an asset.
    # Dollars live here. Shadow prices live on the signal and the momentum book.
    prices = np.zeros((num_steps + 1, num_paths, num_assets))
    
    # Opening book is the allocation weights scaled to the starting total.
    # Without initial_value, S0 is already the per-asset valuation.
    starting = np.asarray(S0, dtype=float)
    if initial_value is not None:
        starting = starting * (float(initial_value) / starting.sum())
    prices[0] = starting

    regime = None
    regime_factors = None
    if regime_drift is not None:
        regime_kwargs = {}
        if transition is not None:
            regime_kwargs["transition"] = transition
        if transition_interval is not None:
            regime_kwargs["transition_interval"] = transition_interval
        # The tilt uses the trigger's SMA Enter length and keeps every close in
        # that window. SMA-exclusion is a portfolio rule. It does not apply here,
        # so the regime average and the sleeve average of the same asset differ.
        window = sma_window
        if window is None and signal is not None and trigger_index is not None:
            window = int(signal.signals[int(trigger_index)].enter)
        regime_kwargs["sma_window"] = SMA_TREND_DAYS if window is None else window
        regime_kwargs["sma_per_std"] = sma_per_std
        regime_kwargs["sma_z_cap"] = sma_z_cap
        if assumptions is not None:
            regime_kwargs["churn_vol_grace"] = assumptions.churn_vol_grace_months
            regime_kwargs["churn_vol_step"] = assumptions.churn_vol_monthly_step
        regime = RegimeBook(
            regime_drift, regime_volatility, num_paths, trigger_index, initial_states, rng=regime_rng, **regime_kwargs
        )
        if regime_states is not None:
            regime_states[0] = regime.states
        # Expansion keeps the historical factor, except DBMF, whose equity
        # correlation is the live monthly one. Churn damps the matrix, and
        # Liquidation pulls every other pair toward 0.85. BTC joins that
        # stress only partway and keeps its historical correlation with cash.
        # DBMF moves against equities in that stress and keeps its historical
        # correlation with cash, gold, bitcoin, and other managed futures.
        # Each path uses the factor for its state.
        if cholesky is not None:
            baseline_factor = np.asarray(cholesky, dtype=float)
            regime_factors = regime_cholesky_factors(baseline_factor @ baseline_factor.T, labels, assumptions)
    else:
        drift_mat = np.broadcast_to(np.asarray(drift, dtype=float), (num_paths, num_assets))
        vol_mat = np.broadcast_to(np.asarray(volatility, dtype=float), (num_paths, num_assets))

    # The book records market growth, not the rebalanced valuation. With no center,
    # the trailing return is compared with the return implied by today's drift.
    momentum = None
    if lambda_mom is not None:
        momentum = MomentumBook(lambda_mom, momentum_center, momentum_lookback, num_paths, num_assets)

    # The lead is shadow prices only. It gives the SMA a full window so the
    # open can be cash or a momentum bucket, and it does not age the regime.
    if signal is not None and signal.uses_momentum_sleeve and not signal._ready:
        if regime is None:
            warm_drift, warm_vol = drift_mat, vol_mat
            warm_cholesky, warm_states = cholesky, None
        else:
            warm_drift, warm_vol = regime.parameters()
            if regime_factors is None:
                warm_cholesky, warm_states = cholesky, None
            else:
                warm_cholesky, warm_states = regime_factors, regime.states
        open_from_total_gbm(
            signal, warm_drift, warm_vol, dt, target_allocation, warm_cholesky, warm_states, momentum
        )
        prices[0] = prices[0].sum(axis=1, keepdims=True) * signal.target_weights(target_allocation)

    # Credit a whole stretch on each trade day. Selection is constant in between,
    # so the daily loop does not tally.
    count_momentum = (
        momentum_days is not None
        and signal is not None
        and signal.uses_momentum_sleeve
    )
    if count_momentum:
        held_since = 0
    
    # Per path, per asset: which band the weight crossed.
    lower_counts = np.zeros((num_paths, num_assets))
    upper_counts = np.zeros((num_paths, num_assets))

    # Holdings change only on a momentum trade, so the mask is rebuilt then.
    track_holding = (
        signal is not None
        and signal.uses_momentum_sleeve
        and (count_momentum or momentum_log is not None)
    )
    cached_holding = signal.holding_mask() if track_holding else None

    # Chronological Time Loop
    for t in range(1, num_steps + 1):
        # Drift and volatility are supplied by the regime book. Rebalancing does not choose them.
        if regime is None:
            step_drift = drift_mat
            step_vol = vol_mat
            step_cholesky = cholesky
            step_states = None
        else:
            step_drift, step_vol = regime.parameters()
            if regime_factors is None:
                step_cholesky = cholesky
                step_states = None
            else:
                step_cholesky = regime_factors
                step_states = regime.states
            if regime_states is not None:
                regime_states[t] = regime.states
        if momentum is not None:
            step_drift = momentum.adjust(step_drift, dt)
        # Band checks and momentum trades are independent moduli of t.
        # A momentum book marks every asset at its market drift, rotates the
        # sleeve, then on a check day may overwrite that result with a band snap.
        # Any other book either swaps unheld sleeves onto cash drift, or, with
        # no signal at all, grows the static targets.
        market_growth = None
        rebalance_day = t % interval == 0
        lower_breach = None
        if signal is not None and signal.uses_momentum_sleeve:
            market_growth = gbm_growth(step_drift, step_vol, dt, step_cholesky, step_states)
            # The trade rotates after this step's growth, so the day belongs to the prior bucket.
            prior_holding = cached_holding
            trade_day = count_momentum and (signal.steps + 1) % signal.trade_interval == 0
            held_mask = prior_holding if trade_day else None
            shifted, decision_day = momentum_shift(prices[t - 1], signal, market_growth, target_allocation)
            if held_mask is not None and decision_day:
                momentum_days += held_mask.astype(np.int64) * (t - held_since)
                held_since = t
            if decision_day and track_holding:
                cached_holding = signal.holding_mask()
            if rebalance_day:
                # The snap is priced off the pre-rotation book. Only alternates
                # with a positive target can cause it, so the sleeve cannot
                # trade the whole path just by winning a ranking.
                grown = prices[t - 1] * market_growth
                step_targets = signal.target_weights(target_allocation)
                trigger = np.broadcast_to(signal.alternate_mask, step_targets.shape).copy()
                trigger &= step_targets > 0
                prices[t], lower_breach, upper_breach = dynamic_rebalance(
                    grown, lower_threshold, upper_threshold, step_targets, trigger
                )
                # A breached path keeps the snapped book. A quiet path keeps the
                # rotation. Both books already use the post-trade selection.
                if decision_day:
                    triggered = (lower_breach | upper_breach).any(axis=1)
                    prices[t] = np.where(triggered[:, None], prices[t], shifted)
            else:
                prices[t] = shifted
            if momentum_log is not None and prior_holding is not None:
                momentum_log += prior_holding * _portfolio_log_growth(prices[t - 1], prices[t])[:, None]
        else:
            if signal is None:
                market_growth = gbm_growth(step_drift, step_vol, dt, step_cholesky, step_states)
                next_prices = prices[t - 1] * market_growth
                step_targets = target_allocation
            else:
                next_prices, step_targets, market_growth = monthly_momentum(
                    prices[t - 1],
                    step_drift,
                    step_vol,
                    dt,
                    signal,
                    target_allocation,
                    step_cholesky,
                    step_states,
                    with_targets=rebalance_day,
                )
            if rebalance_day:
                # A zero-target alternate sits inside a wide band on the sheet.
                # Its weight is zero, and that row cannot force the book to trade.
                trigger = None
                if signal is not None and signal.static_allocation:
                    trigger = np.asarray(step_targets, dtype=float) > 0
                prices[t], lower_breach, upper_breach = dynamic_rebalance(
                    next_prices, lower_threshold, upper_threshold, step_targets, trigger
                )
            else:
                prices[t] = next_prices
            if signal is not None:
                signal.observe(market_growth)

        if momentum is not None:
            momentum.observe(market_growth)

        if lower_breach is not None:
            lower_counts += lower_breach
            upper_counts += upper_breach
        if regime is not None:
            # The trigger compounds market growth of one asset, starting at 1.
            # observe() may draw the next state. That state is recorded on the
            # following day, so today's return keeps the state it was generated with.
            # Churn volatility on the trigger creeps up inside parameters(), from
            # the days already accumulated, not from the day that just finished.
            trigger_growth = None
            if market_growth is not None and regime.trigger_index is not None:
                trigger_growth = np.asarray(market_growth, dtype=float)[:, regime.trigger_index]
            regime.observe(trigger_growth)

    if count_momentum:
        remaining = num_steps - held_since
        if remaining:
            momentum_days += signal.holding_mask().astype(np.int64) * remaining
        
    return prices, lower_counts, upper_counts


REGIME_STATE_NAMES = ((1, "Expansion"), (2, "Churn"), (3, "Liquidation"))


def _finished_spell_years(occupied, state, dt):
    """Years in each visit to `state` that ends before the last step.

    A visit is a run of consecutive steps in `state`. It starts when a path
    enters and ends when the path leaves. The run still underway on the final
    step has not finished, so its length is unknown and it is left out.
    `occupied` is steps by paths, without the opening duplicate row.
    """
    mask = occupied == state
    steps, paths = mask.shape
    if steps == 0 or paths == 0:
        return np.empty(0, dtype=float)
    entered = np.vstack([np.zeros((1, paths), dtype=bool), mask])
    changes = np.diff(entered.astype(np.int8), axis=0)
    start_time, start_path = np.nonzero(changes == 1)
    end_time, end_path = np.nonzero(changes == -1)
    if start_time.size == 0 or end_time.size == 0:
        return np.empty(0, dtype=float)
    start_order = np.lexsort((start_time, start_path))
    end_order = np.lexsort((end_time, end_path))
    start_time = start_time[start_order]
    start_path = start_path[start_order]
    end_time = end_time[end_order]
    start_counts = np.bincount(start_path, minlength=paths)
    end_counts = np.bincount(end_path, minlength=paths)
    start_offsets = np.cumsum(start_counts) - start_counts
    start_rank = np.arange(start_time.size) - np.repeat(start_offsets, start_counts)
    finished = start_rank < np.repeat(end_counts, start_counts)
    if not np.any(finished):
        return np.empty(0, dtype=float)
    return (end_time - start_time[finished]).astype(float) * dt


def summarize_regime_time(regime_states, dt):
    """Years each path spends in Expansion, Churn, and Liquidation.

    Each row after the open is one step of length `dt`. The opening row repeats
    the state already used on the first step, so it is not an extra interval.
    Mean Duration and Median Duration are the length of one finished visit to
    that regime, in years. A visit still open on the last step is not a
    finished market cycle, and a regime with no finished visit has no duration.
    """
    regime_states = np.asarray(regime_states)
    occupied = regime_states[1:] if regime_states.shape[0] > 1 else regime_states[:0]
    if occupied.ndim == 1:
        occupied = occupied[:, None]
    horizon = occupied.shape[0] * dt
    rows = []
    for state, name in REGIME_STATE_NAMES:
        years = (occupied == state).sum(axis=0) * dt
        share = 0.0 if horizon == 0 else float(years.mean() / horizon)
        spells = _finished_spell_years(occupied, state, dt)
        if spells.size:
            mean_duration = float(spells.mean())
            median_duration = float(np.median(spells))
        else:
            mean_duration = np.nan
            median_duration = np.nan
        rows.append({
            "State": name,
            "Mean Years": float(years.mean()) if years.size else 0.0,
            "Median Years": float(np.median(years)) if years.size else 0.0,
            "Share of Time": share,
            "Mean Duration": mean_duration,
            "Median Duration": median_duration,
        })
    return pd.DataFrame(rows)


def summarize_regime_cagr(portfolio_paths, regime_states, dt):
    """CAGR earned while the book was in each regime.

    The return from value[t - 1] to value[t] belongs to regime_states[t].
    The opening row repeats the state already used on the first step, so it is
    not an extra return. Mean CAGR compounds every day spent in the regime, so a
    path that only visits for a day cannot dominate it. Median CAGR is the median
    of each visiting path's own annualized return.
    """
    portfolio_paths = np.asarray(portfolio_paths, dtype=float)
    regime_states = np.asarray(regime_states)
    if portfolio_paths.ndim == 1:
        portfolio_paths = portfolio_paths[:, None]
    if regime_states.ndim == 1:
        regime_states = regime_states[:, None]

    steps = 0
    if portfolio_paths.shape[0] > 1 and regime_states.shape[0] > 1:
        steps = min(portfolio_paths.shape[0], regime_states.shape[0]) - 1
    occupied = regime_states[1:steps + 1]
    previous = portfolio_paths[:steps]
    current = portfolio_paths[1:steps + 1]
    growth = np.ones((steps, portfolio_paths.shape[1]))
    if steps:
        traded = previous > 0
        growth[traded] = current[traded] / previous[traded]
    log_growth = np.zeros_like(growth) if steps == 0 else np.log(growth)

    rows = []
    for state, name in REGIME_STATE_NAMES:
        if steps == 0:
            rows.append({"State": name, "Mean CAGR": np.nan, "Median CAGR": np.nan})
            continue
        in_state = occupied == state
        years = in_state.sum(axis=0) * dt
        visited = years > 0
        log_sum = np.where(in_state, log_growth, 0.0).sum(axis=0)
        cagr = np.full(years.shape, np.nan)
        cagr[visited] = np.exp(log_sum[visited] / years[visited]) - 1
        if visited.any():
            total_years = float(years[visited].sum())
            mean_cagr = float(np.exp(log_sum[visited].sum() / total_years) - 1)
            median_cagr = float(np.median(cagr[visited]))
        else:
            mean_cagr = np.nan
            median_cagr = np.nan
        rows.append({
            "State": name,
            "Mean CAGR": mean_cagr,
            "Median CAGR": median_cagr,
        })
    return pd.DataFrame(rows)


def summarize_momentum_time(momentum_days, labels, dt, include=None, log_growth=None):
    """Years each path spends holding each momentum bucket.

    `momentum_days` counts trading days and has shape (num_paths, num_assets).
    Each counted day is one step of length `dt`. `include` drops other sleeves.
    Shares use every counted day, including columns left out of the table.
    `log_growth` is the summed log portfolio return earned on those days, one
    column per bucket. Mean CAGR compounds every day the bucket was held.
    Median CAGR is the median of each visiting path's own annualized return.
    A bucket that was never held has no CAGR.
    """
    momentum_days = np.asarray(momentum_days)
    if momentum_days.ndim == 1:
        momentum_days = momentum_days[None, :]
    if log_growth is not None:
        log_growth = np.asarray(log_growth, dtype=float)
        if log_growth.ndim == 1:
            log_growth = log_growth[None, :]
    labels = list(labels)
    if include is None:
        columns = range(momentum_days.shape[1])
    else:
        columns = np.flatnonzero(np.asarray(include, dtype=bool))
    per_path = momentum_days.sum(axis=1)
    horizon = float(per_path.mean() * dt) if per_path.size else 0.0
    rows = []
    for index in columns:
        years = momentum_days[:, index] * dt
        share = 0.0 if horizon == 0 else float(years.mean() / horizon)
        row = {
            "Bucket": labels[int(index)],
            "Mean Years": float(years.mean()) if years.size else 0.0,
            "Median Years": float(np.median(years)) if years.size else 0.0,
            "Share of Time": share,
        }
        if log_growth is not None:
            row["Mean CAGR"], row["Median CAGR"] = _cagr_from_logs(log_growth[:, int(index)], years)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_momentum_log(log_growth, book):
    """Collapse member log returns the same way `candidate_momentum_days` collapses days.

    Members of a bucket are held together, so the first member's compounded
    portfolio return is the bucket's return. Solo names and cash stay on their own row.
    """
    log_growth = np.asarray(log_growth, dtype=float)
    if log_growth.ndim == 1:
        log_growth = log_growth[None, :]
    if not book.has_buckets:
        return log_growth
    columns = [log_growth[:, members[0]] for _key, members in book._candidates]
    if book.cash_index is not None:
        columns.append(log_growth[:, book.cash_index])
    return np.column_stack(columns)


def ending_cagr(ending_value, initial_value, years):
    """Annual growth that compounds `initial_value` into `ending_value` over `years`.

    This is the rate on the valuation printout. A horizon of zero years or a
    non-positive starting value has no rate.
    """
    ending_value = float(ending_value)
    initial_value = float(initial_value)
    years = float(years)
    if years <= 0 or initial_value <= 0 or not np.isfinite(ending_value) or ending_value < 0:
        return np.nan
    return (ending_value / initial_value) ** (1.0 / years) - 1


def path_max_drawdown(portfolio_paths):
    """Deepest peak-to-trough loss on each portfolio path.

    The peak is the highest value at or before a step. The loss at that step
    is the value divided by the peak, minus one. A book that never falls is
    zero. A book that falls is negative, and a later recovery does not erase
    the loss. A peak that is not positive has no loss. One number comes back
    per path.
    """
    portfolio_paths = np.asarray(portfolio_paths, dtype=float)
    if portfolio_paths.ndim == 1:
        portfolio_paths = portfolio_paths[:, None]
    peak = np.maximum.accumulate(portfolio_paths, axis=0)
    drawdown = np.zeros(portfolio_paths.shape, dtype=float)
    traded = peak > 0
    drawdown[traded] = portfolio_paths[traded] / peak[traded] - 1.0
    return drawdown.min(axis=0)


def path_sharpe_ratio(portfolio_paths, years, risk_free=0.0):
    """Annualized Sharpe ratio of each path against a constant cash drift.

    Each step's simple return is the portfolio's growth minus one. `risk_free`
    is the continuous drift cash compounds at, so the hurdle for a step of
    length dt is exp(risk_free * dt) - 1. The ratio is the mean excess return
    divided by its sample standard deviation, multiplied by the square root of
    the number of steps in one year. A path with fewer than two returns, or
    with no variation, has no ratio. A step whose previous value is not
    positive contributes no further change.
    """
    portfolio_paths = np.asarray(portfolio_paths, dtype=float)
    if portfolio_paths.ndim == 1:
        portfolio_paths = portfolio_paths[:, None]
    years = float(years)
    risk_free = float(risk_free)
    if portfolio_paths.shape[0] < 3 or years <= 0 or not np.isfinite(risk_free):
        return np.full(portfolio_paths.shape[1], np.nan)
    previous = portfolio_paths[:-1]
    current = portfolio_paths[1:]
    simple = np.zeros(previous.shape, dtype=float)
    traded = previous > 0
    simple[traded] = current[traded] / previous[traded] - 1.0
    steps = simple.shape[0]
    excess = simple - (np.exp(risk_free * years / steps) - 1.0)
    center = excess.mean(axis=0)
    deviation = excess.std(axis=0, ddof=1)
    sharpe = np.full(center.shape, np.nan)
    varying = deviation > 0
    sharpe[varying] = center[varying] / deviation[varying] * np.sqrt(steps / years)
    return sharpe


def _finite_rank(values, rank):
    """Mean, median, or percentile, ignoring entries that are not finite.

    `rank` is None for the mean and "median" for the median. Any other value
    is a percentile from 0 to 100. A set with no finite entries has no rank.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    if rank is None:
        return float(finite.mean())
    if rank == "median":
        return float(np.median(finite))
    return float(np.percentile(finite, rank))


def summarize_valuation_metrics(portfolio_paths, years, risk_free=0.0):
    """Ending value, CAGR, max drawdown, and Sharpe ratio for each final statistic.

    Every path starts from the same book. CAGR is the rate that compounds that
    start into the statistic's ending value. Max drawdown and the Sharpe ratio
    use the same rank on each path: the 90th percentile is the mild upside,
    the mean and median are the center, and the 5th percentile is the deep
    floor. Drawdowns are negative. The Sharpe ratio is annualized excess
    return over `risk_free` per unit of variation. The starting value has no
    horizon, so its CAGR, max drawdown, and Sharpe ratio are left blank.
    A path with no variation is left out of the Sharpe ranks.
    """
    portfolio_paths = np.asarray(portfolio_paths, dtype=float)
    initial = float(portfolio_paths[0, 0])
    ending = portfolio_paths[-1]
    drawdown = path_max_drawdown(portfolio_paths)
    sharpe = path_sharpe_ratio(portfolio_paths, years, risk_free)
    rows = [
        ("Initial Portfolio Value", initial, False, np.nan, np.nan),
        (
            "Upside 90th Percentile",
            float(np.percentile(ending, 90)),
            True,
            float(np.percentile(drawdown, 90)),
            _finite_rank(sharpe, 90),
        ),
        (
            "Expected Mean Ending Value",
            float(ending.mean()),
            True,
            float(drawdown.mean()),
            _finite_rank(sharpe, None),
        ),
        (
            "Median Ending Value",
            float(np.median(ending)),
            True,
            float(np.median(drawdown)),
            _finite_rank(sharpe, "median"),
        ),
        (
            "95% Confidence Portfolio Floor (5th%)",
            float(np.percentile(ending, 5)),
            True,
            float(np.percentile(drawdown, 5)),
            _finite_rank(sharpe, 5),
        ),
    ]
    return pd.DataFrame(
        [
            {
                "Metric": name,
                "Value": value,
                "CAGR": ending_cagr(value, initial, years) if annualize else np.nan,
                "Max Drawdown": loss,
                "Sharpe Ratio": ratio,
            }
            for name, value, annualize, loss, ratio in rows
        ]
    )


def _cagr_from_logs(log_sum, years):
    """Annualize summed log returns. Paths with no time in the bucket are left out.

    Mean CAGR compounds the pooled log return over the pooled years, so a
    path that held the bucket for one day cannot dominate it. Median CAGR is
    the median of the per-path rates. Both are empty when nobody visited.
    """
    visited = np.asarray(years) > 0
    if not np.any(visited):
        return np.nan, np.nan
    log_sum = np.asarray(log_sum, dtype=float)
    path_cagr = np.exp(log_sum[visited] / np.asarray(years)[visited]) - 1
    mean_cagr = float(np.exp(log_sum[visited].sum() / float(np.asarray(years)[visited].sum())) - 1)
    return mean_cagr, float(np.median(path_cagr))


def _portfolio_log_growth(previous, current):
    """Log change in total portfolio value from one book to the next.

    Each path is summed across assets first, so a rotation that only moves
    dollars inside the book contributes nothing. A non-positive previous
    total is treated as no change rather than as a logarithm of zero.
    """
    previous_total = np.asarray(previous, dtype=float).sum(axis=1)
    current_total = np.asarray(current, dtype=float).sum(axis=1)
    growth = np.ones(previous_total.shape)
    traded = previous_total > 0
    growth[traded] = current_total[traded] / previous_total[traded]
    return np.log(growth)


def candidate_momentum_days(momentum_days, book, labels):
    """Collapse members that share a bucket into one column.

    Days are recorded on every member, and those members are held together, so the
    bucket's time is the longest count among them. Solo names stay on their own row.
    Without buckets the array and labels are returned unchanged.
    """
    momentum_days = np.asarray(momentum_days)
    if momentum_days.ndim == 1:
        momentum_days = momentum_days[None, :]
    labels = list(labels)
    if not book.has_buckets:
        return momentum_days, labels
    columns = []
    names = []
    for key, members in book._candidates:
        columns.append(momentum_days[:, list(members)].max(axis=1))
        names.append(key[1] if key[0] == "bucket" else labels[members[0]])
    if book.cash_index is not None:
        columns.append(momentum_days[:, book.cash_index])
        names.append(labels[book.cash_index])
    return np.column_stack(columns), names


def summarize_trigger_buckets(lower_counts, upper_counts, labels, include=None):
    """Collapse path-level band breaches into one row per allocation bucket.

    `include` drops names that cannot cause a band rebalance. Shares are
    recomputed on the rows that remain.
    """
    labels = list(labels)
    if include is not None:
        columns = np.flatnonzero(np.asarray(include, dtype=bool))
        lower_counts = np.asarray(lower_counts)[:, columns]
        upper_counts = np.asarray(upper_counts)[:, columns]
        labels = [labels[int(index)] for index in columns]
    lower_counts = np.asarray(lower_counts)
    upper_counts = np.asarray(upper_counts)
    num_paths = lower_counts.shape[0]
    lower_total = lower_counts.sum(axis=0)
    upper_total = upper_counts.sum(axis=0)
    total = lower_total + upper_total
    total_triggers = total.sum()
    share = np.zeros(total.shape, dtype=float) if total_triggers == 0 else total / total_triggers

    return pd.DataFrame({
        "Bucket": labels,
        "Lower Band Triggers": lower_total.astype(int),
        "Upper Band Triggers": upper_total.astype(int),
        "Total Triggers": total.astype(int),
        "Share of Triggers": share,
        "Paths with a Trigger": ((lower_counts + upper_counts) > 0).sum(axis=0).astype(int),
        "Mean Triggers per Path": total / num_paths,
    })


def trading_rebalance_interval(dynamic_rebalance):
    """Daily band checks when requested, otherwise once per trading month.

    The month is 21 simulation steps, the same cadence as the default regime
    draw. It is not the momentum trade interval, which is 25.
    """
    return 1 if dynamic_rebalance else TRADING_DAYS_PER_MONTH


def horizon_years(frame, default=10):
    """Years to simulate, from the first Time to Maturity cell.

    The cell is a length in years. Rows after the first are not read.
    A missing column, or a blank, non-numeric, or non-positive first cell,
    keeps `default`. That default is the length `main` used before this
    column was read, including the dummy frame used when the workbook is missing.
    """
    columns = getattr(frame, "columns", ())
    if frame is None or "Time to Maturity" not in columns or len(frame) == 0:
        return float(default)
    value = frame["Time to Maturity"].iloc[0]
    if pd.isna(value):
        return float(default)
    try:
        years = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(years) or years <= 0:
        return float(default)
    return years


def parse_args(argv=None):
    """Command line for `main`. Horizon, path count, and starting value are not flags.

    The horizon is the first Time to Maturity on the model sheet, in years.

    `--sheet-name` chooses the model. `--dynamic-rebalance` checks bands every
    day instead of every 21. `--top_num` is how many momentum candidates share
    the sleeve. `--beat_cash` is on unless `--no-beat_cash` is passed. The
    hurdle compares the percent gap over the SMA with the annual cash drift.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the portfolio simulation for one Excel sheet.")
    parser.add_argument("--sheet-name", default="Asset Parameters", help="Model sheet, for example 'Sector Momentum Playbook'")
    parser.add_argument(
        "--dynamic-rebalance",
        action="store_true",
        help="Check allocation bands every trading day. The default checks once per trading month.",
    )
    parser.add_argument(
        "--top_num",
        type=int,
        default=MOMENTUM_PICKS,
        help="How many momentum buckets or solo stocks to hold. Each receives an equal share of the sleeve.",
    )
    parser.add_argument(
        "-beat_cash",
        "--beat_cash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Hold a momentum name only when its gap beats the cash rate. "
            "Pass --no-beat_cash to hold any name that sits above its SMA."
        ),
    )
    return parser.parse_args(argv)


def main(sheet_name="Asset Parameters", dynamic_rebalance=False, top_num=MOMENTUM_PICKS, beat_cash=True):
    """Run one 5,000-path simulation for the model on `sheet_name`.

    The length, in years, is the first Time to Maturity on that sheet.
    A missing cell keeps the 10-year default.

    The workbook is the source of the drift, the fee, the stress block, and
    the momentum coefficients. Code defaults apply only where the sheet has
    no cell.

    The model sheet supplies labels, bands, targets, the horizon, SMA windows,
    asset type, and buckets. A literal Net Expense Ratio on that sheet replaces the fee
    table. A formula keeps the Net Expense Ratios lookup.

    Asset State Behaviors supplies a drift and a volatility for each regime,
    and the momentum lambda and center the run actually uses. Bitcoin is not
    rewritten after that read.

    The transition sheet supplies the Markov matrix, the SMA tilt, and the
    stress block: churn scale, liquidation blend, bitcoin and DBMF crisis
    correlations, churn-volatility creep, and the cash-flight ticker list.
    Cash flight still copies SGOV's gross State 3 row onto those tickers
    after fees, because that is the number the sheet names.
    """
    # Assumes Excel formatting matches your image (header on Row 1)
    # If Row 1 is a title card, keep header=1, otherwise use header=0
    workbook = "portfolio_gbm_calculator.xlsx"
    try:
        df = pd.read_excel(workbook, header=0, sheet_name=sheet_name)
    except FileNotFoundError:
        # Fallback dummy dataframe matching your image exactly for testing
        print("No Excel file found, using dummy dataframe")
        data = {
            "Target Allocation": [0.30, 0.15, 0.10, 0.10, 0.10, 0.10],
            "Lower Band": [0.27, 0.12, 0.07, 0.07, 0.02, 0.02],
            "Upper Band": [0.37, 0.22, 0.17, 0.17, 0.14, 0.14],
            "Drift (Annual)": [0.10, 0.11, 0.07, 0.35, 0.09, 0.22],
            "Volatility (Annual)": [0.15, 0.18, 0.10, 0.60, 0.18, 0.50]
        }
        df = pd.DataFrame(data)
    print("DataFrame:")
    print("--------------------------------")
    print(df)
    print("--------------------------------")
    
    # Pull values directly from your spreadsheet format.
    # Drift and volatility come from Asset State Behaviors, one column pair per regime.
    # The first Time to Maturity is the horizon in years. Later cells are ignored.
    S0 = df["Target Allocation"].values
    T = horizon_years(df)
    dt = 1/252
    num_paths = 5000
    rebalance_interval = trading_rebalance_interval(dynamic_rebalance)
    initial_value = 171655
    labels = df["Ticker"].tolist() if "Ticker" in df.columns else [f"Asset {i + 1}" for i in range(len(S0))]
    behavior = pd.read_excel(workbook, sheet_name="Asset State Behaviors")
    assumptions = load_simulation_assumptions(workbook)
    # The behavior row is the drift. Cash flight, when the sheet names tickers,
    # is applied after fees so it can still see SGOV's gross State 3 cell.
    regime_drift, regime_volatility = align_regime_parameters(behavior, labels, cash_flight=())
    expense_ratios = overlay_model_expenses(load_expense_ratios(workbook), workbook, sheet_name)
    regime_drift = np.vstack([
        apply_expense_ratios(regime_drift[state], labels, expense_ratios) for state in range(3)
    ])
    regime_drift, regime_volatility = apply_cash_flight(
        regime_drift, regime_volatility, labels, behavior, assumptions.cash_flight
    )
    # Bull, Churn, and Bear. The sheet says how many trading days pass between draws.
    transition, transition_interval, sma_per_std, sma_z_cap = load_transition_matrix(workbook)
    drift = regime_drift[1]
    volatility = regime_volatility[1]
    cash_row = _row_for_label({str(row["Ticker"]).strip(): row for _, row in behavior.iterrows()}, CASH_TICKER)
    cash_drift = float(cash_row["State 3 Drift"])
    cash_volatility = float(cash_row["State 3 Volatility"])
    tracked_index = df["Tracked Index"] if "Tracked Index" in df.columns else [None] * len(labels)
    asset_type = df["Asset Type"] if "Asset Type" in df.columns else None
    sma_exit = df["SMA Exit"] if "SMA Exit" in df.columns else [None] * len(labels)
    sma_enter = df["SMA Enter"] if "SMA Enter" in df.columns else [None] * len(labels)
    sma_exclusion = df["SMA-exclusion"] if "SMA-exclusion" in df.columns else [0] * len(labels)
    buckets = df["Bucket"] if "Bucket" in df.columns else None
    signals = resolve_asset_signals(
        labels, sma_exit, sma_enter, tracked_index, asset_type=asset_type, sma_exclusion=sma_exclusion, bucket=buckets
    )
    cash_index = cash_asset_index(labels)
    lambda_mom, _historical_trailing = overlay_sheet_momentum(
        *load_momentum_drift(labels), labels, behavior
    )
    signal = SignalBook(
        signals, num_paths, cash_drift, cash_volatility, cash_index=cash_index, top_num=top_num, beat_cash=beat_cash
    )
    cholesky = simulation_cholesky(labels, sheet_name)
    
    lower_threshold = df["Lower Band"].values
    upper_threshold = df["Upper Band"].values
    target_allocation = df["Target Allocation"].values
    
    # Run the corrected 3D matrix simulation
    # 1. Run the core simulation to get the 3D price matrix
    regime_states = np.zeros((int(T / dt) + 1, num_paths), dtype=int)
    momentum_days = np.zeros((num_paths, len(labels)), dtype=np.int64)
    momentum_log = np.zeros((num_paths, len(labels)))
    simulated_matrix, lower_counts, upper_counts = simulate_with_rebalance(
        S0, drift, volatility, T, dt, num_paths, 
        lower_threshold, upper_threshold, target_allocation,
        initial_value=initial_value,
        cholesky=cholesky,
        regime_drift=regime_drift,
        regime_volatility=regime_volatility,
        trigger_index=_regime_trigger(labels),
        regime_states=regime_states,
        signal=signal,
        labels=labels,
        transition=transition,
        transition_interval=transition_interval,
        rebalance_interval=rebalance_interval,
        momentum_days=momentum_days,
        momentum_log=momentum_log,
        lambda_mom=lambda_mom,
        sma_per_std=sma_per_std,
        sma_z_cap=sma_z_cap,
        assumptions=assumptions,
    )

    # 2. Compress the Assets dimension to find the Total Portfolio Value at every step.
    # Shape changes from (steps + 1, paths, assets) to (steps + 1, paths).
    # A 20-year run is 5,041 rows. A 10-year run is 2,521.
    portfolio_paths = np.sum(simulated_matrix, axis=2)

    # 3. Calculate cross-sectional statistics over all 5,000 parallel universes
    mean_portfolio_growth = np.mean(portfolio_paths, axis=1)    # Average performance path
    median_portfolio_growth = np.median(portfolio_paths, axis=1)  # 50th percentile path
    downside_5th_percentile = np.percentile(portfolio_paths, 5, axis=1)  # 5th percentile (Risk Floor)

    # 4. Wrap the statistical summaries into a clean Pandas DataFrame for analysis
    time_axis = np.linspace(0, T, int(T / dt) + 1)
    df_growth_summary = pd.DataFrame({
        'Mean Path': mean_portfolio_growth,
        'Median Path': median_portfolio_growth,
        '5th Pct Downside': downside_5th_percentile
    }, index=time_axis)
    df_growth_summary.index.name = 'Time (Years)'

    regime_time = summarize_regime_time(regime_states, dt)
    regime_time_display = regime_time.copy()
    regime_time_display["Share of Time"] = regime_time_display["Share of Time"].map(lambda value: f"{value:.1%}")
    for column in ("Mean Duration", "Median Duration"):
        regime_time_display[column] = regime_time_display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:,.2f}"
        )
    print("\n--- TIME IN EACH REGIME ---")
    print(regime_time_display.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    if signal.uses_momentum_sleeve:
        momentum_buckets = signal.momentum_mask.copy()
        if signal.cash_index is not None:
            momentum_buckets[signal.cash_index] = True
        if signal.has_buckets:
            collapsed, names = candidate_momentum_days(momentum_days, signal, labels)
            collapsed_log = candidate_momentum_log(momentum_log, signal)
            momentum_time = summarize_momentum_time(collapsed, names, dt, log_growth=collapsed_log)
        else:
            momentum_time = summarize_momentum_time(
                momentum_days, labels, dt, momentum_buckets, log_growth=momentum_log
            )
        momentum_time_display = momentum_time.copy()
        momentum_time_display["Share of Time"] = momentum_time_display["Share of Time"].map(
            lambda value: f"{value:.1%}"
        )
        for column in ("Mean CAGR", "Median CAGR"):
            momentum_time_display[column] = momentum_time_display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )
        print("\n--- TIME IN EACH MOMENTUM BUCKET ---")
        print(momentum_time_display.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    regime_cagr = summarize_regime_cagr(portfolio_paths, regime_states, dt)
    regime_cagr_display = regime_cagr.copy()
    for column in ("Mean CAGR", "Median CAGR"):
        regime_cagr_display[column] = regime_cagr_display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2%}"
        )
    print("\n--- AVERAGE CAGR BY REGIME ---")
    print(regime_cagr_display.to_string(index=False))

    band_names = ~signal.momentum_mask if signal.uses_momentum_sleeve else None
    trigger_summary = summarize_trigger_buckets(lower_counts, upper_counts, labels, include=band_names)
    print("\n--- BUCKETS THAT CAUSED A REBALANCE ---")
    print(trigger_summary.sort_values("Total Triggers", ascending=False).to_string(index=False))

    # The hurdle is the cash yield the book can hold: State 3 drift after the SGOV fee.
    cash_yield = float(apply_expense_ratios(np.array([cash_drift]), [CASH_TICKER], expense_ratios)[0])
    metrics = summarize_valuation_metrics(portfolio_paths, T, risk_free=cash_yield)
    metrics_display = metrics.copy()
    metrics_display["Value"] = metrics_display["Value"].map(lambda value: f"${value:.2f}")
    for column in ("CAGR", "Max Drawdown"):
        metrics_display[column] = metrics_display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2%}"
        )
    metrics_display["Sharpe Ratio"] = metrics_display["Sharpe Ratio"].map(
        lambda value: "" if pd.isna(value) else f"{value:.2f}"
    )
    print(f"\n--- FINAL PORTFOLIO VALUATION METRICS ({float(T):g}-YEARS) ---")
    print(metrics_display.to_string(index=False))


def _regime_trigger(labels):
    """VOO drives the regime. A sheet without VOO uses the first asset that is not cash."""
    try:
        return trigger_index(labels)
    except KeyError:
        for index, label in enumerate(labels):
            if str(label).strip() != CASH_TICKER:
                print(f"Regime trigger: {label}")
                return index
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args.sheet_name, dynamic_rebalance=args.dynamic_rebalance, top_num=args.top_num, beat_cash=args.beat_cash)


