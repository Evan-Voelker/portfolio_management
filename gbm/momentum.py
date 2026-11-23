from pathlib import Path

import numpy as np
import pandas as pd

from gbm.asset_simulation_matrix import market_ticker
from gbm.regime import CASH_TICKER, TRADING_DAYS_PER_MONTH, _row_for_label

# One trading year. The historical slope uses the same twelve-month gap.
MOMENTUM_LOOKBACK_MONTHS = 12
MOMENTUM_LOOKBACK_DAYS = MOMENTUM_LOOKBACK_MONTHS * TRADING_DAYS_PER_MONTH
# The add-on, not the regime drift, stops here so a large trailing return cannot run away.
MOMENTUM_DRIFT_FLOOR = -0.40
MOMENTUM_DRIFT_CAP = 0.35
# One enormous year, such as early Bitcoin, would otherwise dominate the slope.
TRAILING_RETURN_FLOOR = -0.80
TRAILING_RETURN_CAP = 2.0
MIN_OBSERVATIONS = 24
MOMENTUM_DRIFT_PATH = Path(__file__).resolve().with_name("momentum_drift.csv")


def active_drift(base, trailing, lambda_mom, center, floor=MOMENTUM_DRIFT_FLOOR, cap=MOMENTUM_DRIFT_CAP):
    """Regime drift plus a clipped momentum add-on.

    The add-on is lambda times the trailing simple return minus `center`.
    A trailing return sitting on `center` leaves `base` unchanged, so a
    symmetric gap does not move the average drift. `floor` and `cap` bound
    that add-on. The regime drift itself is left alone.
    """
    base = np.asarray(base, dtype=float)
    gap = np.asarray(trailing, dtype=float) - np.asarray(center, dtype=float)
    adjustment = np.clip(np.asarray(lambda_mom, dtype=float) * gap, floor, cap)
    return base + adjustment


class MomentumBook:
    """Shadow price of each asset, and the drift add-on that price implies.

    The book compounds market growth only. Rebalance and sleeve rotation never
    enter the trailing return. Until a full lookback exists, the drift is the
    regime drift.
    """

    def __init__(self, lambda_mom, center, lookback, num_paths, num_assets):
        self.lambda_mom = np.asarray(lambda_mom, dtype=float).reshape(1, -1)
        if self.lambda_mom.shape != (1, num_assets):
            raise ValueError("momentum coefficients must cover every asset")
        if center is None:
            self.center = None
        else:
            self.center = np.asarray(center, dtype=float).reshape(1, -1)
            if self.center.shape != (1, num_assets):
                raise ValueError("momentum coefficients must cover every asset")
        self.lookback = int(lookback)
        if self.lookback < 1:
            raise ValueError("momentum lookback must be at least 1 trading day")
        self.levels = np.ones((self.lookback + 1, int(num_paths), int(num_assets)))
        self.pos = 0
        self.seen = 0

    def adjust(self, base, dt):
        """Drift for the coming step, from the close already stored.

        With no stored center, the trailing return is measured against the
        simple return the current drift itself would produce over the lookback.
        A path that earns exactly that drift keeps it.
        """
        base = np.asarray(base, dtype=float)
        if self.seen < self.lookback:
            return base
        past = self.levels[(self.pos - self.lookback) % (self.lookback + 1)]
        current = self.levels[self.pos]
        valid = past > 0
        trailing = np.zeros_like(current)
        np.divide(current, past, out=trailing, where=valid)
        center = self._center(base, dt)
        trailing = np.where(valid, trailing - 1.0, center)
        return active_drift(base, trailing, self.lambda_mom, center)

    def _center(self, base, dt):
        if self.center is not None:
            return self.center
        horizon = self.lookback * float(dt)
        return np.exp(base * horizon) - 1.0

    def observe(self, growth):
        """Advance the shadow price by one market growth factor."""
        new_pos = (self.pos + 1) % (self.lookback + 1)
        self.levels[new_pos] = self.levels[self.pos] * np.asarray(growth, dtype=float)
        self.pos = new_pos
        self.seen += 1


def estimate_momentum_drift(prices, lookback_months=MOMENTUM_LOOKBACK_MONTHS, horizon_months=MOMENTUM_LOOKBACK_MONTHS):
    """Slope of the next 12-month return on the past 12-month return, and that past return's mean.

    Lambda is the ordinary-least-squares slope after yearly simple returns
    outside [-80%, +200%] are pulled in, so one extreme year cannot set the
    slope. Center is the mean of the raw trailing returns on those same rows.
    A flat series, or one with fewer than two years of pairs, gets a zero
    slope and a zero center.
    """
    monthly = prices.resample("ME").last()
    trailing = monthly / monthly.shift(lookback_months) - 1.0
    forward = monthly.shift(-horizon_months) / monthly - 1.0
    capped_trailing = trailing.clip(TRAILING_RETURN_FLOOR, TRAILING_RETURN_CAP)
    capped_forward = forward.clip(TRAILING_RETURN_FLOOR, TRAILING_RETURN_CAP)
    rows = []
    for column in monthly.columns:
        paired = pd.DataFrame({
            "trailing": trailing[column],
            "capped_trailing": capped_trailing[column],
            "capped_forward": capped_forward[column],
        }).dropna()
        if len(paired) < MIN_OBSERVATIONS:
            rows.append((str(column), 0.0, 0.0))
            continue
        slope = _slope(paired["capped_trailing"].to_numpy(), paired["capped_forward"].to_numpy())
        center = float(paired["trailing"].mean())
        if not np.isfinite(slope):
            slope = 0.0
        if not np.isfinite(center):
            center = 0.0
        rows.append((str(column), slope, center))
    return pd.DataFrame(rows, columns=["Ticker", "Lambda", "Center"])


def align_momentum_drift(frame, labels):
    """Lambda and center in `labels` order. Cash stays at zero, and an unknown ticker stays at zero.

    The frame is used as written. A behavior sheet can replace any of these
    cells afterward, including bitcoin and DBMF.
    """
    table = frame.copy()
    if "Ticker" not in getattr(table, "columns", []):
        table = table.reset_index()
    if "Ticker" not in table.columns:
        table = table.rename(columns={table.columns[0]: "Ticker"})
    lookup = {}
    for _, row in table.iterrows():
        key = str(row["Ticker"]).strip()
        if key and key.lower() != "nan":
            lookup[key] = row
    lambda_mom = np.zeros(len(labels))
    center = np.zeros(len(labels))
    for index, label in enumerate(labels):
        key = market_ticker(label)
        if key == CASH_TICKER:
            continue
        row = lookup.get(key)
        if row is None:
            continue
        lambda_mom[index] = float(row["Lambda"])
        center[index] = float(row["Center"])
    return lambda_mom, center


def overlay_sheet_momentum(lambda_mom, center, labels, behavior):
    """Replace cache coefficients with Momentum Lambda and Momentum Center from the behavior sheet.

    A blank cell leaves the cache value. Cash stays at zero unless the sheet
    writes a number on that row. A ticker with no behavior row is left alone.
    """
    columns = [name for name in ("Momentum Lambda", "Momentum Center") if name in behavior.columns]
    if not columns:
        return lambda_mom, center
    lookup = {str(row["Ticker"]).strip(): row for _, row in behavior.iterrows()}
    lambda_mom = np.array(lambda_mom, dtype=float, copy=True)
    center = np.array(center, dtype=float, copy=True)
    for index, label in enumerate(labels):
        try:
            row = _row_for_label(lookup, label)
        except KeyError:
            continue
        if "Momentum Lambda" in columns:
            value = _sheet_number(row["Momentum Lambda"])
            if value is not None:
                lambda_mom[index] = value
        if "Momentum Center" in columns:
            value = _sheet_number(row["Momentum Center"])
            if value is not None:
                center[index] = value
    return lambda_mom, center


def _sheet_number(value):
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def load_momentum_drift(labels, path=None, prices=None, download=None):
    """Historical lambda and center for `labels`, cached beside the Cholesky factor.

    Missing risky tickers are estimated from `prices` or from a price download,
    then written back to the cache. SGOV is always zero: its drift is a yield.
    """
    path = MOMENTUM_DRIFT_PATH if path is None else Path(path)
    frame = _read_momentum_drift(path)
    needed = _missing_tickers(labels, frame)
    if needed:
        if prices is None:
            if download is None:
                from gbm.asset_simulation_matrix import download_prices

                download = download_prices
            prices = download(needed)
        estimated = estimate_momentum_drift(prices)
        frame = _merge_momentum_drift(frame, estimated)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    if frame is None:
        frame = pd.DataFrame(columns=["Ticker", "Lambda", "Center"])
    return align_momentum_drift(frame, labels)


def _slope(trailing, forward):
    trailing = np.asarray(trailing, dtype=float)
    forward = np.asarray(forward, dtype=float)
    centered = trailing - trailing.mean()
    variance = float(np.dot(centered, centered))
    if variance <= 1e-18:
        return 0.0
    return float(np.dot(centered, forward - forward.mean()) / variance)


def _missing_tickers(labels, frame):
    known = set() if frame is None else set(frame["Ticker"].astype(str).str.strip())
    needed = []
    for label in labels:
        key = market_ticker(label)
        if key == CASH_TICKER or key in known or key in needed:
            continue
        needed.append(key)
    return needed


def _merge_momentum_drift(saved, estimated):
    estimated = estimated.copy()
    estimated["Ticker"] = estimated["Ticker"].astype(str).str.strip()
    if saved is None or saved.empty:
        return estimated
    current = saved.copy()
    current["Ticker"] = current["Ticker"].astype(str).str.strip()
    current = current.set_index("Ticker")
    incoming = estimated.set_index("Ticker")
    merged = incoming.combine_first(current)
    return merged.reset_index()[["Ticker", "Lambda", "Center"]]


def _read_momentum_drift(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if "Ticker" not in frame.columns or frame.empty:
        return None
    frame["Ticker"] = frame["Ticker"].astype(str).str.strip()
    return frame
