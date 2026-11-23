"""Historical backtest of a model sheet.

The allocation rules are the same ones the Monte Carlo uses: SMA signals,
the momentum sleeve, gated dollars, and allocation bands. The growth is not.
Each session multiplies the book by that day's close divided by the previous
close. There is no regime draw and no correlated shock.

The scored window is the last 10 years unless `--start_date` names the first
scored session. One year of earlier closes warms the averages, and that year
is extended when an SMA window is longer than it. Warmup gains are not part
of the valuation. The book opens at the starting value on the last warmup close.
SGOV, which listed in 2020, is extended backward with the 3-month Treasury
bill index before that listing. VXUS, which listed in 2011, is extended
backward with VGTSX, the 1996 investor class of the same Vanguard fund.
VOO, which listed in 2010, is extended backward with VFINX, the investor
class of the same 500 Index Fund. That series starts before SPY. CSVYX,
which listed in 2009, is extended backward with CSCZX and CSMIX, the older
classes of the same small-cap value fund. TQQQ, which listed in 2010, is
extended backward to 1996 with a daily-reset 3x Nasdaq-100. QQQ is the
total return from 1999, and the Nasdaq-100 price index covers the years
before that. The synthetic path charges TQQQ's expense ratio and finances
the borrowed two dollars at the 3-month bill yield.

    python -m gbm.bt
    python -m gbm.bt --start_date 01/02/2016 --sheet_name "Sector Momentum Playbook"
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gbm.asset_simulation_matrix import WORKBOOK_PATH, download_prices, market_ticker
from gbm.expenses import apply_expense_ratios, load_expense_ratios, overlay_model_expenses
from gbm.gbm_step import (
    _portfolio_log_growth,
    candidate_momentum_days,
    candidate_momentum_log,
    dynamic_rebalance,
    ending_cagr,
    momentum_shift,
    path_max_drawdown,
    path_sharpe_ratio,
    summarize_momentum_time,
    summarize_trigger_buckets,
    trading_rebalance_interval,
)
from gbm.regime import CASH_TICKER, TRADING_DAYS_PER_MONTH, _row_for_label
from gbm.signals import MOMENTUM_PICKS, SignalBook, cash_asset_index, resolve_asset_signals

# Scored horizon when the caller does not name a start date.
BACKTEST_YEARS = 10
# Closes kept before the first scored session so the averages exist at the open.
WARMUP_YEARS = 1
# Same opening book as the Monte Carlo, so a backtest dollar is the same dollar.
INITIAL_VALUE = 171655
DATE_FORMAT = "%m/%d/%Y"


@dataclass
class ModelStrategy:
    """One sheet's labels, bands, targets, and the signal book that trades them."""

    sheet_name: str
    labels: list
    signal: SignalBook
    target_allocation: np.ndarray
    lower_threshold: np.ndarray
    upper_threshold: np.ndarray
    cash_drift: float
    cash_yield: float


@dataclass
class MarketBacktest:
    """One historical path. `dates[0]` is the open. Later rows are scored sessions."""

    valuations: np.ndarray
    dates: pd.DatetimeIndex
    lower_counts: np.ndarray
    upper_counts: np.ndarray
    labels: list
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    warmup_start: pd.Timestamp
    warmup_sessions: int
    momentum_days: np.ndarray = None
    momentum_log: np.ndarray = None


def parse_start_date(text):
    """Parse `mm/dd/yyyy`. Anything else is not a start date."""
    try:
        parsed = pd.to_datetime(str(text).strip(), format=DATE_FORMAT)
    except (TypeError, ValueError):
        raise ValueError("start_date must be mm/dd/yyyy") from None
    if pd.isna(parsed):
        raise ValueError("start_date must be mm/dd/yyyy")
    return pd.Timestamp(parsed).normalize()


def backtest_bounds(start_date=None, as_of=None, years=BACKTEST_YEARS, warmup_years=WARMUP_YEARS):
    """Return the warmup start, the first scored day, and the last day.

    With no start date the scored window is `years` calendar years ending on
    `as_of`. The warmup starts `warmup_years` earlier. A start date replaces
    the beginning of the scored window and still keeps that warmup year.
    """
    as_of = pd.Timestamp.today().normalize() if as_of is None else _as_timestamp(as_of)
    if start_date is None:
        eval_start = as_of - pd.DateOffset(years=int(years))
    elif isinstance(start_date, pd.Timestamp):
        eval_start = pd.Timestamp(start_date).normalize()
    else:
        eval_start = parse_start_date(start_date)
    if eval_start > as_of:
        raise ValueError("start_date is after the end of the backtest")
    warmup_start = eval_start - pd.DateOffset(years=int(warmup_years))
    return warmup_start, eval_start, as_of


def prices_for_labels(frame, labels):
    """One close column per sleeve, in label order. VOO/VIIX uses the VOO column."""
    frame = _naive_prices(frame)
    columns = []
    for label in labels:
        symbol = market_ticker(label)
        if label in frame.columns:
            series = frame[label]
        elif symbol in frame.columns:
            series = frame[symbol]
        else:
            raise KeyError(f"{label} has no closes")
        columns.append(pd.Series(np.asarray(series, dtype=float), index=frame.index, name=label))
    return pd.concat(columns, axis=1)


def slice_history(prices, eval_start, warmup_start, lead_days):
    """Split closes into the warmup sample and the scored sample.

    The warmup is every session in the year before `eval_start`. When that
    year is shorter than `lead_days`, earlier sessions are included until the
    signal's longest average can be filled. The scored sample starts on the
    first session on or after `eval_start`.
    """
    prices = _naive_prices(prices)
    eval_start = _as_timestamp(eval_start)
    warmup_start = _as_timestamp(warmup_start)
    needed = max(int(lead_days), 1)
    prior = prices[prices.index < eval_start]
    scored = prices[prices.index >= eval_start]
    if scored.empty:
        raise ValueError(f"no prices on or after {eval_start.strftime(DATE_FORMAT)}")
    year = prior[prior.index >= warmup_start]
    warmup = year if len(year) >= needed else prior.tail(needed)
    if len(warmup) < needed:
        raise ValueError(
            f"warmup has {len(prior)} closes before {eval_start.strftime(DATE_FORMAT)}, "
            f"and the strategy needs {int(lead_days)}"
        )
    return warmup, scored


def backtest_strategy(
    growth,
    signal,
    target_allocation,
    lower_threshold,
    upper_threshold,
    initial_value=INITIAL_VALUE,
    rebalance_interval=TRADING_DAYS_PER_MONTH,
    warmup_closes=None,
    momentum_days=None,
    momentum_log=None,
):
    """Replay one book on supplied growth factors.

    `growth` is scored sessions by assets. A factor is that session's close
    divided by the previous close. `warmup_closes` are the price levels before
    the open, oldest first. They fill the SMA and, for a momentum sleeve, pick
    the opening holdings. The valuation starts at `initial_value` after that
    warmup, so warmup gains do not enter the path.

    A momentum sleeve marks every name at its own close, then rotates on the
    same 25-session clock the simulation uses. Any other book earns the cash
    close on sleeves the signal has turned off, and the signal still records
    the real close. Bands are checked on `rebalance_interval`. A breach
    replaces the whole book with the post-trade targets.
    """
    growth = np.asarray(growth, dtype=float)
    if growth.ndim != 2:
        raise ValueError("growth must be sessions by assets")
    if not np.isfinite(growth).all() or np.any(growth <= 0):
        raise ValueError("growth factors must be finite and positive")
    interval = int(rebalance_interval)
    if interval < 1:
        raise ValueError("rebalance_interval must be at least 1 trading day")
    if signal is not None and signal.invested.shape[0] != 1:
        raise ValueError("a market backtest runs one path")

    num_steps, num_assets = growth.shape
    valuations = np.zeros((num_steps + 1, 1, num_assets))
    if warmup_closes is not None and signal is not None:
        _warm_signal(signal, warmup_closes)
    if signal is not None and signal.uses_momentum_sleeve:
        if warmup_closes is None:
            raise ValueError("a momentum backtest needs warmup closes")
        signal.start(in_cash=True)
        signal.choose_momentum(target_allocation)
        opening = signal.target_weights(target_allocation)
    else:
        opening = np.asarray(target_allocation, dtype=float)
    valuations[0] = _scaled_weights(opening, initial_value)

    count_momentum = momentum_days is not None and signal is not None and signal.uses_momentum_sleeve
    if count_momentum:
        held_since = 0
    lower_counts = np.zeros((1, num_assets))
    upper_counts = np.zeros((1, num_assets))
    track_holding = signal is not None and signal.uses_momentum_sleeve and (
        count_momentum or momentum_log is not None
    )
    cached_holding = signal.holding_mask() if track_holding else None

    for step in range(1, num_steps + 1):
        market_growth = growth[step - 1].reshape(1, -1)
        rebalance_day = step % interval == 0
        lower_breach = None
        if signal is not None and signal.uses_momentum_sleeve:
            prior_holding = cached_holding
            trade_day = count_momentum and (signal.steps + 1) % signal.trade_interval == 0
            held_mask = prior_holding if trade_day else None
            shifted, decision_day = momentum_shift(valuations[step - 1], signal, market_growth, target_allocation)
            if held_mask is not None and decision_day:
                momentum_days += held_mask.astype(np.int64) * (step - held_since)
                held_since = step
            if decision_day and track_holding:
                cached_holding = signal.holding_mask()
            if rebalance_day:
                grown = valuations[step - 1] * market_growth
                step_targets = signal.target_weights(target_allocation)
                trigger = np.broadcast_to(signal.alternate_mask, step_targets.shape).copy()
                trigger &= step_targets > 0
                valuations[step], lower_breach, upper_breach = dynamic_rebalance(
                    grown, lower_threshold, upper_threshold, step_targets, trigger
                )
                if decision_day:
                    triggered = (lower_breach | upper_breach).any(axis=1)
                    valuations[step] = np.where(triggered[:, None], valuations[step], shifted)
            else:
                valuations[step] = shifted
            if momentum_log is not None and prior_holding is not None:
                momentum_log += prior_holding * _portfolio_log_growth(valuations[step - 1], valuations[step])[:, None]
        else:
            if signal is None:
                next_prices = valuations[step - 1] * market_growth
                step_targets = target_allocation
            else:
                next_prices = valuations[step - 1] * _held_growth(signal, market_growth)
                step_targets = signal.target_weights(target_allocation) if rebalance_day else None
            if rebalance_day:
                trigger = None
                if signal is not None and signal.static_allocation:
                    trigger = np.asarray(step_targets, dtype=float) > 0
                valuations[step], lower_breach, upper_breach = dynamic_rebalance(
                    next_prices, lower_threshold, upper_threshold, step_targets, trigger
                )
            else:
                valuations[step] = next_prices
            if signal is not None:
                signal.observe(market_growth)

        if lower_breach is not None:
            lower_counts += lower_breach
            upper_counts += upper_breach

    if count_momentum:
        remaining = num_steps - held_since
        if remaining:
            momentum_days += signal.holding_mask().astype(np.int64) * remaining
    return valuations, lower_counts, upper_counts


def market_backtest(
    prices,
    labels,
    signal,
    target_allocation,
    lower_threshold,
    upper_threshold,
    start_date=None,
    as_of=None,
    years=BACKTEST_YEARS,
    warmup_years=WARMUP_YEARS,
    initial_value=INITIAL_VALUE,
    rebalance_interval=TRADING_DAYS_PER_MONTH,
):
    """Run the strategy on a close history and return the one scored path.

    `prices` is a date-indexed frame of market closes. Columns may be sleeve
    labels or the ticker those sleeves follow. The default window ends on the
    last close, or on `as_of` when that is earlier, and scores the prior
    `years`. `start_date` (`mm/dd/yyyy`) replaces that start.
    """
    labels = list(labels)
    aligned = prices_for_labels(prices, labels)
    if aligned.empty:
        raise ValueError("no closes for the model")
    sample_end = aligned.index.max() if as_of is None else min(_as_timestamp(as_of), aligned.index.max())
    warmup_start, eval_start, eval_end = backtest_bounds(
        start_date, as_of=sample_end, years=years, warmup_years=warmup_years
    )
    aligned = aligned[aligned.index <= eval_end]
    lead_days = 0 if signal is None else int(signal.lead_days)
    _require_history(aligned, eval_start, lead_days)
    warmup, scored = slice_history(aligned.dropna(how="any"), eval_start, warmup_start, lead_days)
    bridge = pd.concat([warmup.tail(1), scored])
    growth = bridge.iloc[1:].to_numpy(dtype=float) / bridge.iloc[:-1].to_numpy(dtype=float)
    momentum_days = None
    momentum_log = None
    if signal is not None and signal.uses_momentum_sleeve:
        momentum_days = np.zeros((1, len(labels)), dtype=np.int64)
        momentum_log = np.zeros((1, len(labels)))
    valuations, lower_counts, upper_counts = backtest_strategy(
        growth,
        signal,
        target_allocation,
        lower_threshold,
        upper_threshold,
        initial_value=initial_value,
        rebalance_interval=rebalance_interval,
        warmup_closes=None if signal is None else warmup.to_numpy(dtype=float),
        momentum_days=momentum_days,
        momentum_log=momentum_log,
    )
    return MarketBacktest(
        valuations=valuations,
        dates=pd.DatetimeIndex(bridge.index),
        lower_counts=lower_counts,
        upper_counts=upper_counts,
        labels=labels,
        eval_start=eval_start,
        eval_end=pd.Timestamp(scored.index[-1]).normalize(),
        warmup_start=warmup_start,
        warmup_sessions=int(len(warmup)),
        momentum_days=momentum_days,
        momentum_log=momentum_log,
    )


def load_model(sheet_name="Asset Parameters", top_num=MOMENTUM_PICKS, beat_cash=True, workbook=None):
    """Read the strategy on `sheet_name`. Drift and volatility are not used.

    The cash hurdle is SGOV's State 3 drift, the same comparison the Monte
    Carlo makes. The fee-adjusted cash yield is kept for the Sharpe ratio.
    Historical closes already include the fund's expenses, so the backtest
    does not subtract them again.
    """
    workbook = WORKBOOK_PATH if workbook is None else workbook
    frame = pd.read_excel(workbook, header=0, sheet_name=sheet_name)
    labels = frame["Ticker"].tolist() if "Ticker" in frame.columns else [f"Asset {i + 1}" for i in range(len(frame))]
    behavior = pd.read_excel(workbook, sheet_name="Asset State Behaviors")
    lookup = {str(row["Ticker"]).strip(): row for _, row in behavior.iterrows()}
    cash_row = _row_for_label(lookup, CASH_TICKER)
    cash_drift = float(cash_row["State 3 Drift"])
    expense_ratios = overlay_model_expenses(load_expense_ratios(workbook), workbook, sheet_name)
    cash_yield = float(apply_expense_ratios(np.array([cash_drift]), [CASH_TICKER], expense_ratios)[0])
    tracked_index = frame["Tracked Index"] if "Tracked Index" in frame.columns else [None] * len(labels)
    asset_type = frame["Asset Type"] if "Asset Type" in frame.columns else None
    sma_exit = frame["SMA Exit"] if "SMA Exit" in frame.columns else [None] * len(labels)
    sma_enter = frame["SMA Enter"] if "SMA Enter" in frame.columns else [None] * len(labels)
    sma_exclusion = frame["SMA-exclusion"] if "SMA-exclusion" in frame.columns else [0] * len(labels)
    buckets = frame["Bucket"] if "Bucket" in frame.columns else None
    signals = resolve_asset_signals(
        labels, sma_exit, sma_enter, tracked_index, asset_type=asset_type, sma_exclusion=sma_exclusion, bucket=buckets
    )
    signal = SignalBook(
        signals,
        num_paths=1,
        cash_drift=cash_drift,
        cash_volatility=float(cash_row["State 3 Volatility"]),
        cash_index=cash_asset_index(labels),
        top_num=top_num,
        beat_cash=beat_cash,
    )
    return ModelStrategy(
        sheet_name=sheet_name,
        labels=labels,
        signal=signal,
        target_allocation=np.asarray(frame["Target Allocation"], dtype=float),
        lower_threshold=np.asarray(frame["Lower Band"], dtype=float),
        upper_threshold=np.asarray(frame["Upper Band"], dtype=float),
        cash_drift=cash_drift,
        cash_yield=cash_yield,
    )


def parse_args(argv=None):
    """Command line for the historical backtest.

    `--sheet_name` chooses the model, the same sheets the Monte Carlo reads.
    `--start_date` is the first scored session, `mm/dd/yyyy`. Omitting it
    scores the last 10 years. The other flags change the strategy, not the
    price history.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Backtest one model sheet on historical closes.")
    parser.add_argument(
        "--sheet_name",
        "--sheet-name",
        default="Asset Parameters",
        dest="sheet_name",
        help="Model sheet, for example 'Sector Momentum Playbook'.",
    )
    parser.add_argument(
        "--start_date",
        "--start-date",
        default=None,
        dest="start_date",
        help="First scored session, mm/dd/yyyy. The default is 10 years before the latest close.",
    )
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


def main(sheet_name="Asset Parameters", start_date=None, dynamic_rebalance=False, top_num=MOMENTUM_PICKS, beat_cash=True, prices=None, as_of=None):
    """Backtest `sheet_name` and print the one historical path.

    Prices are downloaded from the warmup start, with extra room for an SMA
    that is longer than that year. A caller can pass `prices` to skip the
    download. `as_of` caps the sample; the default end is the latest close.
    """
    model = load_model(sheet_name, top_num=top_num, beat_cash=beat_cash)
    if prices is None:
        symbols = []
        for label in model.labels:
            key = market_ticker(label)
            if key not in symbols:
                symbols.append(key)
        today = pd.Timestamp.today().normalize() if as_of is None else _as_timestamp(as_of)
        warmup_start, _eval_start, _end = backtest_bounds(start_date, as_of=today)
        download_start = warmup_start - pd.Timedelta(days=max(int(model.signal.lead_days), 1) * 2)
        print(
            f"Downloading closes for {', '.join(symbols)} from {pd.Timestamp(download_start).strftime(DATE_FORMAT)}...",
            flush=True,
        )
        prices = download_prices(symbols, start=download_start)
    result = market_backtest(
        prices,
        model.labels,
        model.signal,
        model.target_allocation,
        model.lower_threshold,
        model.upper_threshold,
        start_date=start_date,
        as_of=as_of,
        initial_value=INITIAL_VALUE,
        rebalance_interval=trading_rebalance_interval(dynamic_rebalance),
    )
    _print_backtest(model, result)
    return result


def _print_backtest(model, result):
    """Print the window, the sleeve, the bands, and the one path's valuation."""
    years = (len(result.dates) - 1) / float(TRADING_DAYS_PER_MONTH * 12)
    print(f"\nMarket backtest: {model.sheet_name}")
    print(
        f"Scored: {result.eval_start.strftime(DATE_FORMAT)} through {result.eval_end.strftime(DATE_FORMAT)} "
        f"({len(result.dates) - 1} sessions)"
    )
    print(
        f"Warmup: {result.warmup_start.strftime(DATE_FORMAT)} "
        f"({result.warmup_sessions} sessions, not in the valuation)"
    )
    signal = model.signal
    if signal.uses_momentum_sleeve and result.momentum_days is not None:
        if signal.has_buckets:
            collapsed, names = candidate_momentum_days(result.momentum_days, signal, result.labels)
            collapsed_log = candidate_momentum_log(result.momentum_log, signal)
            momentum_time = summarize_momentum_time(collapsed, names, 1 / 252, log_growth=collapsed_log)
        else:
            buckets = signal.momentum_mask.copy()
            if signal.cash_index is not None:
                buckets[signal.cash_index] = True
            momentum_time = summarize_momentum_time(
                result.momentum_days, result.labels, 1 / 252, buckets, log_growth=result.momentum_log
            )
        display = momentum_time.copy()
        display["Share of Time"] = display["Share of Time"].map(lambda value: f"{value:.1%}")
        for column in ("Mean CAGR", "Median CAGR"):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
        print("\n--- TIME IN EACH MOMENTUM BUCKET ---")
        print(display.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    band_names = ~signal.momentum_mask if signal.uses_momentum_sleeve else None
    triggers = summarize_trigger_buckets(result.lower_counts, result.upper_counts, result.labels, include=band_names)
    print("\n--- BUCKETS THAT CAUSED A REBALANCE ---")
    print(triggers.sort_values("Total Triggers", ascending=False).to_string(index=False))

    portfolio = result.valuations.sum(axis=2)
    ending = float(portfolio[-1, 0])
    initial = float(portfolio[0, 0])
    drawdown = float(path_max_drawdown(portfolio)[0])
    sharpe = float(path_sharpe_ratio(portfolio, years, model.cash_yield)[0])
    rows = pd.DataFrame(
        [
            {"Metric": "Initial Portfolio Value", "Value": initial},
            {"Metric": "Ending Portfolio Value", "Value": ending},
            {"Metric": "CAGR", "Value": ending_cagr(ending, initial, years)},
            {"Metric": "Max Drawdown", "Value": drawdown},
            {"Metric": "Sharpe Ratio", "Value": sharpe},
        ]
    )
    def show(metric, value):
        if metric in ("Initial Portfolio Value", "Ending Portfolio Value"):
            return f"${value:,.2f}"
        if metric == "Sharpe Ratio":
            return "" if pd.isna(value) else f"{value:.2f}"
        return "" if pd.isna(value) else f"{value:.2%}"

    rows["Value"] = [show(metric, value) for metric, value in zip(rows["Metric"], rows["Value"])]
    print("\n--- MARKET BACKTEST VALUATION ---")
    print(rows.to_string(index=False))


def _warm_signal(signal, closes):
    """Play historical closes into the signal, oldest first.

    The first close seeds the ring. Each later close is applied as a growth
    factor, so the enter and exit state at the end of the warmup is the state
    the first scored session inherits. This does not advance the trade clock.
    """
    closes = np.asarray(closes, dtype=float)
    if closes.ndim != 2:
        raise ValueError("warmup closes must be days by assets")
    if closes.shape[0] < 1 or closes.shape[1] != signal.invested.shape[1]:
        raise ValueError("warmup closes do not match the book")
    if not np.isfinite(closes).all() or np.any(closes <= 0):
        raise ValueError("warmup closes must be positive")
    signal.load_history(closes[:1])
    if closes.shape[0] == 1:
        return
    growth = closes[1:] / closes[:-1]
    for step in growth:
        signal.observe(step.reshape(1, -1))


def _held_growth(signal, market_growth):
    """Real closes where the signal is in, and the cash close everywhere else.

    An all-alternate book holds the sheet and earns each asset's own close.
    Without a cash asset there is nowhere to park, so every sleeve keeps its
    own close.
    """
    if signal.static_allocation or signal.cash_index is None:
        return market_growth
    held = signal.held()
    cash = market_growth[:, signal.cash_index][:, None]
    return np.where(held, market_growth, cash)


def _scaled_weights(weights, initial_value):
    """Scale a weight row so the book opens at `initial_value`."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    total = weights.sum(axis=1, keepdims=True)
    if np.any(total <= 0) or not np.isfinite(total).all():
        raise ValueError("opening weights must be positive")
    return weights * (float(initial_value) / total)


def _require_history(prices, eval_start, lead_days):
    """Name the listing that leaves the warmup short of the scored start."""
    needed = max(int(lead_days), 1)
    firsts = []
    for label in prices.columns:
        started = prices[label].first_valid_index()
        if started is None or pd.isna(prices[label]).all():
            raise ValueError(f"{label} has no closes")
        firsts.append((pd.Timestamp(started).normalize(), label))
    started, limiter = max(firsts)
    complete = prices.dropna(how="any")
    prior = int((complete.index < eval_start).sum()) if not complete.empty else 0
    if prior >= needed:
        return
    earliest = ""
    if len(complete) > needed:
        earliest = (
            f" The earliest --start_date is {pd.Timestamp(complete.index[needed]).strftime(DATE_FORMAT)}."
        )
    raise ValueError(
        f"{limiter} history starts {started.strftime(DATE_FORMAT)}, so only {prior} closes "
        f"fall before {eval_start.strftime(DATE_FORMAT)}. The strategy needs {needed}.{earliest}"
    )


def _as_timestamp(value):
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 10 and text[2] == "/" and text[5] == "/":
            return parse_start_date(text)
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.normalize().tz_localize(None)
    return stamp.normalize()


def _naive_prices(frame):
    frame = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.normalize().tz_localize(None)
    frame.index = index.normalize()
    return frame.sort_index()


if __name__ == "__main__":
    import sys

    args = parse_args()
    try:
        main(
            args.sheet_name,
            start_date=args.start_date,
            dynamic_rebalance=args.dynamic_rebalance,
            top_num=args.top_num,
            beat_cash=args.beat_cash,
        )
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None
