import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gbm.gbm_step import simulate_with_rebalance, total_gbm
from gbm.momentum import (
    MOMENTUM_DRIFT_CAP,
    MOMENTUM_DRIFT_FLOOR,
    MomentumBook,
    active_drift,
    align_momentum_drift,
    estimate_momentum_drift,
    load_momentum_drift,
)
from gbm.signals import SignalBook, resolve_asset_signals


DT = 1 / 252


def _prices_with_slope(slope, intercept, months=96, seed=0):
    """Month-end prices whose next 12-month return is intercept + slope * trailing."""
    dates = pd.date_range("2010-01-31", periods=months, freq="ME")
    prices = np.empty(months)
    prices[:12] = 100.0
    first = np.random.default_rng(seed).uniform(-0.15, 0.25, size=12)
    for month in range(12, 24):
        prices[month] = prices[month - 12] * (1.0 + first[month - 12])
    for month in range(24, months):
        trailing = prices[month - 12] / prices[month - 24] - 1.0
        prices[month] = prices[month - 12] * (1.0 + intercept + slope * trailing)
    return pd.DataFrame({"VOO": prices, "SGOV": np.full(months, 100.0)}, index=dates)


class ActiveDriftTest(unittest.TestCase):
    def test_a_trailing_return_on_its_center_keeps_the_base_drift(self):
        base = np.array([0.50, -0.75])
        center = np.array([0.12, 0.04])

        adjusted = active_drift(base, center, np.array([0.80, 1.20]), center)

        np.testing.assert_allclose(adjusted, base)

    def test_symmetric_gaps_average_back_to_the_base_drift(self):
        base = np.array([0.10, 0.10])
        center = np.array([0.12, 0.12])
        trailing = np.array([0.22, 0.02])

        adjusted = active_drift(base, trailing, np.array([0.50, 0.50]), center)

        np.testing.assert_allclose(adjusted, [0.15, 0.05])
        self.assertAlmostEqual(float(adjusted.mean()), 0.10)

    def test_the_add_on_is_clipped_and_the_base_drift_is_not(self):
        base = np.array([0.50])

        capped = active_drift(base, np.array([2.0]), np.array([1.0]), np.array([0.0]))
        floored = active_drift(base, np.array([-2.0]), np.array([1.0]), np.array([0.0]))

        np.testing.assert_allclose(capped, base + MOMENTUM_DRIFT_CAP)
        np.testing.assert_allclose(floored, base + MOMENTUM_DRIFT_FLOOR)


class MomentumHistoryTest(unittest.TestCase):
    def test_month_end_prices_recover_the_trailing_return_slope_and_its_mean(self):
        prices = _prices_with_slope(0.40, 0.01)
        estimated = estimate_momentum_drift(prices).set_index("Ticker")
        monthly = prices.resample("ME").last()
        trailing = monthly / monthly.shift(12) - 1.0
        forward = monthly.shift(-12) / monthly - 1.0
        paired = pd.DataFrame({"trailing": trailing["VOO"], "forward": forward["VOO"]}).dropna()

        self.assertAlmostEqual(estimated.loc["VOO", "Lambda"], 0.40, places=8)
        self.assertAlmostEqual(estimated.loc["VOO", "Center"], float(paired["trailing"].mean()), places=8)

    def test_one_extreme_year_cannot_set_the_slope(self):
        prices = _prices_with_slope(0.20, 0.01)
        prices.iloc[48, 0] = prices.iloc[47, 0] * 40.0

        estimated = estimate_momentum_drift(prices).set_index("Ticker")

        self.assertLess(abs(estimated.loc["VOO", "Lambda"]), 1.0)

    def test_a_flat_history_has_no_momentum_slope(self):
        dates = pd.date_range("2012-01-31", periods=80, freq="ME")
        prices = pd.DataFrame({"DBMF": np.full(len(dates), 50.0)}, index=dates)

        estimated = estimate_momentum_drift(prices).set_index("Ticker")

        self.assertEqual(estimated.loc["DBMF", "Lambda"], 0.0)
        self.assertEqual(estimated.loc["DBMF", "Center"], 0.0)

    def test_loader_maps_a_sleeve_label_and_leaves_cash_at_zero(self):
        prices = _prices_with_slope(0.40, 0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "momentum_drift.csv"
            lambda_mom, center = load_momentum_drift(
                ["VOO/VIIX", "SGOV"], path=path, prices=prices
            )

        np.testing.assert_allclose(lambda_mom, [0.40, 0.0])
        self.assertGreater(center[0], 0.0)
        self.assertEqual(center[1], 0.0)

    def test_loader_asks_for_a_missing_ticker_and_then_reads_the_saved_file(self):
        prices = _prices_with_slope(0.25, 0.02)
        requested = []

        def download(tickers):
            requested.append(list(tickers))
            return prices

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "momentum_drift.csv"
            load_momentum_drift(["VOO"], path=path, download=download)
            requested.clear()
            lambda_mom, _center = load_momentum_drift(["VOO"], path=path, download=download)

        self.assertEqual(requested, [])
        self.assertAlmostEqual(lambda_mom[0], 0.25, places=8)


class MomentumPricePathTest(unittest.TestCase):
    def test_shadow_prices_keep_the_base_drift_until_the_lookback_is_full(self):
        lookback = 4
        mu = np.array([0.20, 0.00])
        book = MomentumBook(np.array([0.50, 0.00]), np.zeros(2), lookback, num_paths=1, num_assets=2)
        drift = np.broadcast_to(mu, (1, 2)).copy()

        closes = total_gbm(drift, np.zeros_like(drift), DT, lookback + 1, momentum=book)

        expected = np.exp(mu * np.arange(1, lookback + 1)[:, None] * DT)
        np.testing.assert_allclose(closes[:lookback, 0, :], expected)
        trailing = closes[lookback - 1, 0, :] - 1.0
        active = active_drift(mu, trailing, np.array([0.50, 0.00]), np.zeros(2))
        np.testing.assert_allclose(closes[lookback, 0, :], closes[lookback - 1, 0, :] * np.exp(active * DT))
        np.testing.assert_allclose(closes[:, 0, 1], 1.0)

    def test_rebalance_jumps_do_not_become_trailing_returns(self):
        lookback = 2
        mu = np.array([0.30, 0.00])
        lam = np.ones(2)
        target = np.array([0.50, 0.50])
        starting = np.array([0.80, 0.20])
        prices, _lower, _upper = simulate_with_rebalance(
            S0=starting,
            drift=mu,
            volatility=np.zeros(2),
            T=3 * DT,
            dt=DT,
            num_paths=1,
            lower_threshold=np.zeros(2),
            upper_threshold=np.array([0.60, 0.30]),
            target_allocation=target,
            rebalance_interval=1,
            lambda_mom=lam,
            momentum_center=np.zeros(2),
            momentum_lookback=lookback,
        )

        growth = np.exp(mu * DT)
        total = float(starting @ growth)
        total *= float(target @ growth)
        trailing = np.exp(mu * lookback * DT) - 1.0
        active = active_drift(mu, trailing, lam, np.zeros(2))
        total *= float(target @ np.exp(active * DT))

        np.testing.assert_allclose(prices[3].sum(), total)
        np.testing.assert_allclose(prices[3, 0], total * target)

    def test_a_cash_sleeve_does_not_earn_the_equity_add_on(self):
        labels = ["VOO", "SGOV"]
        signals = resolve_asset_signals(
            labels,
            sma_exit=[1, 1],
            sma_enter=[1, 1],
            asset_type=["Momentum", "Momentum"],
        )
        book = SignalBook(
            signals,
            num_paths=1,
            cash_drift=0.04,
            cash_volatility=0.0,
            cash_index=1,
            trade_interval=100,
        )
        book.start(in_cash=True)
        cash = 0.04

        prices, _lower, _upper = simulate_with_rebalance(
            S0=np.array([0.0, 1.0]),
            drift=np.array([0.50, cash]),
            volatility=np.zeros(2),
            T=3 * DT,
            dt=DT,
            num_paths=1,
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            target_allocation=np.array([0.0, 1.0]),
            signal=book,
            lambda_mom=np.array([1.0, 0.0]),
            momentum_center=np.zeros(2),
            momentum_lookback=2,
        )

        np.testing.assert_allclose(prices[:, 0, 1], np.exp(cash * np.arange(4) * DT))
        np.testing.assert_allclose(prices[:, 0, 0], 0.0)

    def test_holding_the_asset_earns_the_add_on_after_the_lookback(self):
        labels = ["VOO", "SGOV"]
        signals = resolve_asset_signals(
            labels,
            sma_exit=[1, 1],
            sma_enter=[1, 1],
            asset_type=["Momentum", "Momentum"],
        )
        book = SignalBook(
            signals,
            num_paths=1,
            cash_drift=0.04,
            cash_volatility=0.0,
            cash_index=1,
            trade_interval=100,
        )
        book.start(in_cash=False, asset=0)
        mu = 0.20
        lookback = 2

        prices, _lower, _upper = simulate_with_rebalance(
            S0=np.array([1.0, 0.0]),
            drift=np.array([mu, 0.04]),
            volatility=np.zeros(2),
            T=(lookback + 1) * DT,
            dt=DT,
            num_paths=1,
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            target_allocation=np.array([1.0, 0.0]),
            signal=book,
            rebalance_interval=100,
            lambda_mom=np.array([1.0, 0.0]),
            momentum_center=np.zeros(2),
            momentum_lookback=lookback,
        )

        trailed = np.exp(mu * lookback * DT)
        active = active_drift(np.array([mu]), np.array([trailed - 1.0]), np.array([1.0]), np.array([0.0]))
        expected = trailed * np.exp(float(active[0]) * DT)
        np.testing.assert_allclose(prices[lookback + 1, 0, 0], expected)
        self.assertGreater(prices[lookback + 1, 0, 0], np.exp(mu * (lookback + 1) * DT))

    def test_zero_lambda_matches_a_plain_price_path(self):
        arguments = dict(
            S0=np.array([0.60, 0.40]),
            drift=np.array([0.12, 0.05]),
            volatility=np.array([0.18, 0.10]),
            T=6 * DT,
            dt=DT,
            num_paths=2,
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            target_allocation=np.array([0.60, 0.40]),
            rebalance_interval=100,
        )
        np.random.seed(7)
        plain, _, _ = simulate_with_rebalance(**arguments)
        np.random.seed(7)
        centered, _, _ = simulate_with_rebalance(
            **arguments,
            lambda_mom=np.zeros(2),
            momentum_center=np.array([0.11, 0.04]),
            momentum_lookback=3,
        )

        np.testing.assert_allclose(centered, plain)

    def test_a_path_that_earns_its_drift_keeps_that_drift(self):
        arguments = dict(
            S0=np.array([1.0, 1.0]),
            drift=np.array([0.18, -0.05]),
            volatility=np.zeros(2),
            T=8 * DT,
            dt=DT,
            num_paths=1,
            lower_threshold=np.zeros(2),
            upper_threshold=np.full(2, 2.0),
            target_allocation=np.array([0.50, 0.50]),
            rebalance_interval=100,
        )
        plain, _, _ = simulate_with_rebalance(**arguments)
        active, _, _ = simulate_with_rebalance(
            **arguments,
            lambda_mom=np.array([0.80, -0.40]),
            momentum_lookback=3,
        )

        np.testing.assert_allclose(active, plain)

    def test_saved_coefficients_line_up_with_labels(self):
        frame = pd.DataFrame(
            {"Ticker": ["VOO", "BTC"], "Lambda": [0.20, -0.10], "Center": [0.11, 0.40]}
        )

        lambda_mom, center = align_momentum_drift(frame, ["VOO/VIIX", "SGOV", "BTC"])

        np.testing.assert_allclose(lambda_mom, [0.20, 0.0, -0.10])
        np.testing.assert_allclose(center, [0.11, 0.0, 0.40])


if __name__ == "__main__":
    unittest.main()
