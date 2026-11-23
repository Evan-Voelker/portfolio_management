import numpy as np
import pandas as pd

from gbm.regime import (
    STATE_BEAR_TRAIL,
    STATE_BULL_TRAIL,
    STATE_COLUMNS,
    STATE_TRAIL_MONTHS,
    estimate_state_parameters,
    scale_state_parameters,
)

# ProShares UltraPro QQQ. The fund resets to three times the Nasdaq-100 each day.
# The expense ratio is the one this book charges. Financing on the borrowed
# two dollars is separate and is not in this fee.
TQQQ_LEVERAGE = 3.0
TQQQ_EXPENSE = 0.0086
TRADING_DAYS = 252


def leveraged_drift(drift, volatility, financing, leverage=TQQQ_LEVERAGE, expense=TQQQ_EXPENSE):
    """Arithmetic drift and volatility of a daily-reset leveraged fund.

    `drift` and `volatility` are the underlying's GBM parameters, the same
    ones `gbm_growth` consumes. Volatility scales with leverage. Drift scales
    with leverage and then pays the financing rate on the borrowed
    `(leverage - 1)` dollars plus the expense ratio. The half-variance term
    inside the growth step is the daily-reset decay, so this drift does not
    subtract it again. Correlation is unchanged: a positive multiple of a
    return, less a financing rate, has the underlying's correlation.
    """
    drift = np.asarray(drift, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    financing = np.asarray(financing, dtype=float)
    leveraged = float(leverage) * drift - (float(leverage) - 1.0) * financing - float(expense)
    return leveraged, np.abs(float(leverage)) * volatility


def regime_parameters(asset, market, scenario, trail_months=STATE_TRAIL_MONTHS, bull_trail=STATE_BULL_TRAIL, bear_trail=STATE_BEAR_TRAIL):
    """Scenario drift and daily volatility of a fund inside the market's three states.

    Drift uses the monthly estimator and is scaled into `scenario`, the same
    way the other sleeves are expressed in VOO's behavior-sheet units.
    Volatility is the annualized standard deviation of daily log returns in
    those same months, scaled by the same scenario. A daily GBM step needs
    that daily volatility. Monthly closes hide the crash days a 3x fund
    actually delivers.
    """
    scaled = scale_state_parameters(
        estimate_state_parameters(asset, market, trail_months, bull_trail, bear_trail),
        estimate_state_parameters(market, market, trail_months, bull_trail, bear_trail),
        scenario,
    )
    asset_vol = _daily_state_volatility(asset, market, trail_months, bull_trail, bear_trail)
    market_vol = _daily_state_volatility(market, market, trail_months, bull_trail, bear_trail)
    for _drift_name, volatility_name in STATE_COLUMNS:
        market_sigma = market_vol[volatility_name]
        if market_sigma == 0:
            raise ValueError(f"{volatility_name} cannot scale from a zero market volatility")
        scaled[volatility_name] = float(scenario[volatility_name]) * asset_vol[volatility_name] / market_sigma
    return scaled


def _daily_state_volatility(asset, market, trail_months, bull_trail, bear_trail):
    asset = pd.Series(asset, dtype=float).dropna()
    market = pd.Series(market, dtype=float).dropna()
    paired = pd.concat({"asset": asset, "market": market}, axis=1, sort=True).dropna()
    monthly_market = paired["market"].resample("ME").last().dropna()
    trail = monthly_market / monthly_market.shift(int(trail_months)) - 1.0
    month_return = monthly_market.pct_change()
    labeled = pd.DataFrame({"trail": trail, "market": month_return}).dropna()
    bull = (labeled["trail"] >= float(bull_trail)) & (labeled["market"] > 0)
    bear = (labeled["trail"] <= float(bear_trail)) & (labeled["market"] < 0)
    states = {
        "State 1 Volatility": bull,
        "State 2 Volatility": ~bull & ~bear,
        "State 3 Volatility": bear,
    }
    daily_log = np.log(paired["asset"]).diff()
    months = daily_log.index.to_period("M")
    label_by_month = pd.Series(index=labeled.index.to_period("M"), dtype=object)
    for name, mask in states.items():
        label_by_month.loc[labeled.index[mask].to_period("M")] = name
    parameters = {}
    for name in states:
        sample = daily_log[months.map(label_by_month) == name].dropna()
        if len(sample) < 2:
            raise ValueError(f"{name} needs at least 2 daily closes")
        parameters[name] = float(sample.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return parameters
