import unittest

import numpy as np
import pandas as pd

from gbm.gbm_step import (
    candidate_momentum_days,
    candidate_momentum_log,
    dynamic_rebalance,
    ending_cagr,
    path_max_drawdown,
    path_sharpe_ratio,
    simulate_with_rebalance,
    summarize_momentum_time,
    summarize_regime_cagr,
    summarize_regime_time,
    summarize_trigger_buckets,
    summarize_valuation_metrics,
    total_gbm,
)
from gbm.regime import CHURN, EXPANSION
from gbm.signals import SignalBook, resolve_asset_signals


class RebalanceValuationTest(unittest.TestCase):
    def test_trigger_sets_each_asset_to_target_share_of_portfolio_value(self):
        # Holdings sum to 2, with weights 0.75 and 0.25. Only the first weight
        # breaches, and zero drift/vol keeps the pre-trigger value exact.
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([1.50, 0.50]),
            drift=np.array([0.0, 0.0]),
            volatility=np.array([0.0, 0.0]),
            T=2 / 252,
            dt=1 / 252,
            num_paths=3,
            lower_threshold=np.array([0.50, 0.20]),
            upper_threshold=np.array([0.70, 0.50]),
            target_allocation=np.array([0.60, 0.40]),
            rebalance_interval=1,
        )

        portfolio_value = prices[0].sum(axis=1)
        expected = portfolio_value[:, None] * np.array([0.60, 0.40])

        np.testing.assert_allclose(prices[1], expected)
        np.testing.assert_allclose(prices[1].sum(axis=1), portfolio_value)
        np.testing.assert_array_equal(lower_counts + upper_counts, np.tile([1, 0], (3, 1)))
        np.testing.assert_array_equal(upper_counts, np.tile([1, 0], (3, 1)))
        np.testing.assert_array_equal(lower_counts, np.zeros((3, 2)))
        # Targets sit inside the bands, so the next step does not trade again.
        np.testing.assert_allclose(prices[2], prices[1])

    def test_rebalance_interval_of_one_still_snaps_on_the_first_day(self):
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([1.50, 0.50]),
            drift=np.array([0.0, 0.0]),
            volatility=np.array([0.0, 0.0]),
            T=1 / 252,
            dt=1 / 252,
            num_paths=2,
            lower_threshold=np.array([0.50, 0.20]),
            upper_threshold=np.array([0.70, 0.50]),
            target_allocation=np.array([0.60, 0.40]),
            rebalance_interval=1,
        )

        portfolio_value = prices[0].sum(axis=1)
        np.testing.assert_allclose(prices[1], portfolio_value[:, None] * np.array([0.60, 0.40]))
        np.testing.assert_array_equal(upper_counts, np.tile([1, 0], (2, 1)))
        np.testing.assert_array_equal(lower_counts, np.zeros((2, 2)))

    def test_longer_rebalance_interval_keeps_the_grown_book_between_checks(self):
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([1.50, 0.50]),
            drift=np.array([0.0, 0.0]),
            volatility=np.array([0.0, 0.0]),
            T=2 / 252,
            dt=1 / 252,
            num_paths=2,
            lower_threshold=np.array([0.50, 0.20]),
            upper_threshold=np.array([0.70, 0.50]),
            target_allocation=np.array([0.60, 0.40]),
            rebalance_interval=2,
        )

        np.testing.assert_allclose(prices[1], prices[0])
        portfolio_value = prices[0].sum(axis=1)
        np.testing.assert_allclose(prices[2], portfolio_value[:, None] * np.array([0.60, 0.40]))
        np.testing.assert_array_equal(upper_counts, np.tile([1, 0], (2, 1)))
        np.testing.assert_array_equal(lower_counts, np.zeros((2, 2)))

    def test_bands_are_checked_once_per_trading_month_by_default(self):
        # The opening book is outside the bands. The default check leaves it
        # there until the 21st trading day, then snaps once.
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([1.50, 0.50]),
            drift=np.zeros(2),
            volatility=np.zeros(2),
            T=21 / 252,
            dt=1 / 252,
            num_paths=2,
            lower_threshold=np.array([0.50, 0.20]),
            upper_threshold=np.array([0.70, 0.50]),
            target_allocation=np.array([0.60, 0.40]),
        )

        np.testing.assert_allclose(prices[20], prices[0])
        portfolio_value = prices[0].sum(axis=1)
        np.testing.assert_allclose(prices[21], portfolio_value[:, None] * np.array([0.60, 0.40]))
        np.testing.assert_array_equal(upper_counts, np.tile([1, 0], (2, 1)))
        np.testing.assert_array_equal(lower_counts, np.zeros((2, 2)))

    def test_omitted_interval_matches_an_explicit_trading_month(self):
        from gbm.regime import TRADING_DAYS_PER_MONTH

        kwargs = dict(
            S0=np.array([1.5, 0.5]),
            drift=np.zeros(2),
            volatility=np.zeros(2),
            T=21 / 252,
            dt=1 / 252,
            num_paths=2,
            lower_threshold=np.array([0.50, 0.20]),
            upper_threshold=np.array([0.70, 0.50]),
            target_allocation=np.array([0.60, 0.40]),
        )
        default_prices, default_lower, default_upper = simulate_with_rebalance(**kwargs)
        month_prices, month_lower, month_upper = simulate_with_rebalance(
            **kwargs, rebalance_interval=TRADING_DAYS_PER_MONTH
        )

        np.testing.assert_allclose(default_prices, month_prices)
        np.testing.assert_array_equal(default_lower, month_lower)
        np.testing.assert_array_equal(default_upper, month_upper)
        # A daily check would have snapped on day 1. The month check has not.
        np.testing.assert_allclose(default_prices[1], default_prices[0])

    def test_dynamic_rebalance_flag_selects_a_daily_check(self):
        from gbm.gbm_step import parse_args, trading_rebalance_interval
        from gbm.regime import TRADING_DAYS_PER_MONTH

        daily = parse_args(["--dynamic-rebalance"])
        monthly = parse_args([])
        named = parse_args(["--sheet-name", "Sector Momentum Playbook", "--dynamic-rebalance"])

        self.assertTrue(daily.dynamic_rebalance)
        self.assertFalse(monthly.dynamic_rebalance)
        self.assertEqual(monthly.sheet_name, "Asset Parameters")
        self.assertEqual(named.sheet_name, "Sector Momentum Playbook")
        self.assertTrue(named.dynamic_rebalance)
        self.assertEqual(trading_rebalance_interval(True), 1)
        self.assertEqual(trading_rebalance_interval(False), TRADING_DAYS_PER_MONTH)

    def test_top_num_flag_sets_how_many_momentum_names_are_held(self):
        from gbm.gbm_step import parse_args
        from gbm.signals import MOMENTUM_PICKS

        self.assertEqual(parse_args([]).top_num, MOMENTUM_PICKS)
        self.assertEqual(parse_args(["--top_num", "2"]).top_num, 2)
        self.assertEqual(parse_args(["--sheet-name", "Sector Momentum Playbook", "--top_num", "1"]).top_num, 1)

    def test_beat_cash_flag_switches_the_momentum_hurdle(self):
        from gbm.gbm_step import parse_args

        self.assertTrue(parse_args([]).beat_cash)
        self.assertTrue(parse_args(["-beat_cash"]).beat_cash)
        self.assertTrue(parse_args(["--beat_cash"]).beat_cash)
        self.assertFalse(parse_args(["--no-beat_cash"]).beat_cash)
        self.assertFalse(parse_args(["--sheet-name", "Sector Momentum Playbook", "--no-beat_cash"]).beat_cash)

    def test_rebalance_interval_must_be_at_least_one_trading_day(self):
        with self.assertRaises(ValueError):
            simulate_with_rebalance(
                S0=np.array([1.0]),
                drift=np.array([0.0]),
                volatility=np.array([0.0]),
                T=1 / 252,
                dt=1 / 252,
                num_paths=1,
                lower_threshold=np.array([0.0]),
                upper_threshold=np.array([1.0]),
                target_allocation=np.array([1.0]),
                rebalance_interval=0,
            )

    def test_weights_inside_bands_keep_the_gbm_valuation(self):
        dt = 1 / 252
        drift = 0.10
        s0 = np.array([0.60, 0.40])
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=s0,
            drift=np.array([drift, drift]),
            volatility=np.array([0.0, 0.0]),
            T=dt,
            dt=dt,
            num_paths=2,
            lower_threshold=np.array([0.50, 0.30]),
            upper_threshold=np.array([0.70, 0.50]),
            target_allocation=np.array([0.60, 0.40]),
        )

        growth = np.exp((drift - 0.5 * 0.0**2) * dt)
        np.testing.assert_allclose(prices[1], np.broadcast_to(s0 * growth, prices[1].shape))
        np.testing.assert_array_equal(lower_counts, np.zeros((2, 2)))
        np.testing.assert_array_equal(upper_counts, np.zeros((2, 2)))

    def test_initial_value_scales_opening_holdings_to_that_total(self):
        dt = 1 / 252
        weights = np.array([0.60, 0.40])
        prices, _lower_counts, _upper_counts = simulate_with_rebalance(
            S0=weights,
            drift=np.array([0.0, 0.0]),
            volatility=np.array([0.0, 0.0]),
            T=dt,
            dt=dt,
            num_paths=2,
            lower_threshold=np.array([0.50, 0.30]),
            upper_threshold=np.array([0.70, 0.50]),
            target_allocation=weights,
            initial_value=171655,
        )

        np.testing.assert_allclose(prices[0], np.broadcast_to(171655 * weights, prices[0].shape))
        np.testing.assert_allclose(prices[0].sum(axis=1), np.full(2, 171655))

    def test_simulation_records_the_bucket_and_band_that_breached(self):
        _prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([0.80, 0.20]),
            drift=np.array([0.0, 0.0]),
            volatility=np.array([0.0, 0.0]),
            T=1 / 252,
            dt=1 / 252,
            num_paths=2,
            lower_threshold=np.array([0.50, 0.25]),
            upper_threshold=np.array([0.70, 0.60]),
            target_allocation=np.array([0.60, 0.40]),
            rebalance_interval=1,
        )

        np.testing.assert_array_equal(upper_counts, np.tile([1, 0], (2, 1)))
        np.testing.assert_array_equal(lower_counts, np.tile([0, 1], (2, 1)))

    def test_dynamic_rebalance_trades_each_path_on_its_own_book(self):
        valuations = np.array([
            [75.0, 25.0],
            [60.0, 40.0],
            [80.0, 20.0],
        ])
        targets = np.array([
            [0.60, 0.40],
            [0.10, 0.90],
            [0.25, 0.75],
        ])

        book, lower_breach, upper_breach = dynamic_rebalance(
            valuations,
            lower_threshold=np.array([0.50, 0.20]),
            upper_threshold=np.array([0.70, 0.50]),
            target_weights=targets,
        )

        np.testing.assert_allclose(book[0], [60.0, 40.0])
        np.testing.assert_allclose(book[1], [60.0, 40.0])
        np.testing.assert_allclose(book[2], [25.0, 75.0])
        np.testing.assert_allclose(book.sum(axis=1), valuations.sum(axis=1))
        np.testing.assert_array_equal(upper_breach, [[True, False], [False, False], [True, False]])
        np.testing.assert_array_equal(lower_breach, [[False, False], [False, False], [False, True]])

    def test_summary_counts_lower_and_upper_breaches_by_bucket(self):
        lower = np.array([[1, 0, 2], [0, 0, 1]])
        upper = np.array([[0, 3, 0], [1, 1, 0]])

        summary = summarize_trigger_buckets(lower, upper, ["VOO", "CSVYX", "TQQQ"])

        self.assertEqual(summary["Bucket"].tolist(), ["VOO", "CSVYX", "TQQQ"])
        np.testing.assert_array_equal(summary["Lower Band Triggers"], [1, 0, 3])
        np.testing.assert_array_equal(summary["Upper Band Triggers"], [1, 4, 0])
        np.testing.assert_array_equal(summary["Total Triggers"], [2, 4, 3])
        np.testing.assert_allclose(summary["Share of Triggers"], [2 / 9, 4 / 9, 3 / 9])
        np.testing.assert_array_equal(summary["Paths with a Trigger"], [2, 2, 2])
        np.testing.assert_allclose(summary["Mean Triggers per Path"], [1.0, 2.0, 1.5])

    def test_summary_leaves_momentum_names_out_of_the_rebalance_table(self):
        # XLE can post breaches in the raw counts, and the table still omits it.
        # Shares are recomputed on the names that remain.
        lower = np.array([[1, 0, 2], [0, 0, 1]])
        upper = np.array([[0, 3, 0], [1, 1, 0]])

        summary = summarize_trigger_buckets(
            lower,
            upper,
            ["XLE", "DBMF", "BTC"],
            include=np.array([False, True, True]),
        )

        self.assertEqual(summary["Bucket"].tolist(), ["DBMF", "BTC"])
        np.testing.assert_array_equal(summary["Total Triggers"], [4, 3])
        np.testing.assert_allclose(summary["Share of Triggers"], [4 / 7, 3 / 7])

    def test_summary_converts_regime_steps_into_years_in_each_state(self):
        # The opening row matches the state already used on the first step, so it is not an extra day.
        states = np.array([
            [1, 2],
            [1, 2],
            [1, 2],
            [2, 2],
            [3, 1],
        ])

        summary = summarize_regime_time(states, dt=0.25)

        self.assertEqual(summary["State"].tolist(), ["Expansion", "Churn", "Liquidation"])
        np.testing.assert_allclose(summary["Mean Years"], [0.375, 0.5, 0.125])
        np.testing.assert_allclose(summary["Median Years"], [0.375, 0.5, 0.125])
        np.testing.assert_allclose(summary["Share of Time"], [0.375, 0.5, 0.125])
        # Path 0 finishes a half-year expansion and a quarter-year churn. Its liquidation step is still open.
        # Path 1 finishes a three-quarter churn. Its expansion step is still open.
        np.testing.assert_allclose(summary["Mean Duration"].iloc[:2], [0.5, 0.5])
        np.testing.assert_allclose(summary["Median Duration"].iloc[:2], [0.5, 0.5])
        self.assertTrue(np.isnan(summary["Mean Duration"].iloc[2]))
        self.assertTrue(np.isnan(summary["Median Duration"].iloc[2]))

    def test_regime_duration_measures_one_finished_visit(self):
        # A visit starts when a path enters a regime and ends when it leaves.
        # Path 0 finishes a two-year expansion and a one-year churn. Liquidation is still open.
        # Path 1 finishes a three-year churn. Expansion is still open.
        # Path 2 finishes a one-year expansion, a one-year churn, then stays in expansion.
        # The second expansion has not ended, so only the first counts.
        states = np.array([
            [1, 2, 1],
            [1, 2, 1],
            [1, 2, 2],
            [2, 2, 1],
            [3, 1, 1],
        ])

        summary = summarize_regime_time(states, dt=1.0).set_index("State")

        np.testing.assert_allclose(summary.loc["Expansion", ["Mean Duration", "Median Duration"]], [1.5, 1.5])
        np.testing.assert_allclose(summary.loc["Churn", "Mean Duration"], 5.0 / 3.0)
        np.testing.assert_allclose(summary.loc["Churn", "Median Duration"], 1.0)
        self.assertTrue(np.isnan(summary.loc["Liquidation", "Mean Duration"]))
        self.assertTrue(np.isnan(summary.loc["Liquidation", "Median Duration"]))

    def test_regime_cagr_annualizes_only_the_steps_spent_in_that_state(self):
        # Opening row is the state already used on the first step, so it is not an extra return.
        # Path 0 spends both years in expansion and compounds 10% then 10%.
        # Path 1 spends one year in expansion at 21%, then a flat churn year.
        # Path 2 never enters expansion. Its churn year halves, then liquidation is flat.
        states = np.array([
            [1, 1, 2],
            [1, 1, 2],
            [1, 2, 3],
        ])
        portfolio = np.array([
            [100.0, 100.0, 100.0],
            [110.0, 121.0, 50.0],
            [121.0, 121.0, 50.0],
        ])

        summary = summarize_regime_cagr(portfolio, states, dt=1.0)

        self.assertEqual(summary["State"].tolist(), ["Expansion", "Churn", "Liquidation"])
        # Mean compounds every day in the regime. Expansion has two 10% years and one 21% year.
        # Churn has one flat year and one year that halves.
        np.testing.assert_allclose(summary["Mean CAGR"], [1.21 ** (2.0 / 3.0) - 1, np.sqrt(0.5) - 1, 0.0])
        np.testing.assert_allclose(summary["Median CAGR"], [0.155, -0.25, 0.0])

    def test_regime_cagr_ignores_paths_that_never_enter_the_state(self):
        # Columns are paths. The second path spends the whole window in churn.
        states = np.array([
            [1, 2, 1, 1],
            [1, 2, 1, 1],
            [1, 2, 3, 3],
        ])
        portfolio = np.array([
            [100.0, 100.0, 100.0, 100.0],
            [110.0, 50.0, 130.0, 100.0],
            [121.0, 25.0, 130.0, 100.0],
        ])

        summary = summarize_regime_cagr(portfolio, states, dt=1.0)
        expansion = summary.loc[summary["State"] == "Expansion"].iloc[0]
        churn = summary.loc[summary["State"] == "Churn"].iloc[0]
        liquidation = summary.loc[summary["State"] == "Liquidation"].iloc[0]

        np.testing.assert_allclose(
            expansion["Mean CAGR"],
            np.exp((2 * np.log(1.1) + np.log(1.3)) / 4) - 1,
        )
        np.testing.assert_allclose(expansion["Median CAGR"], 0.10)
        np.testing.assert_allclose(churn["Mean CAGR"], -0.50)
        np.testing.assert_allclose(liquidation["Mean CAGR"], 0.0)
        np.testing.assert_allclose(liquidation["Median CAGR"], 0.0)

    def test_one_day_in_a_regime_cannot_inflate_the_mean_cagr(self):
        # Forty paths lose 4% over a full year in liquidation. One path gains 5% on a single day.
        # Annualizing that day alone is (1.05 ** 252) - 1, about 220,000%, and the arithmetic
        # mean of the path rates is thousands of percent. The reported mean stays near -4%.
        dt = 1 / 252
        long_paths = 40
        steps = 252
        states = np.full((steps + 1, long_paths + 1), 1)
        states[:, :long_paths] = 3
        states[1, long_paths] = 3
        portfolio = np.full((steps + 1, long_paths + 1), 100.0)
        daily = 0.96 ** dt
        portfolio[1:, :long_paths] = 100.0 * np.cumprod(np.full(steps, daily))[:, None]
        portfolio[1:, long_paths] = 105.0

        summary = summarize_regime_cagr(portfolio, states, dt)
        liquidation = summary.loc[summary["State"] == "Liquidation"].iloc[0]

        total_log = long_paths * np.log(0.96) + np.log(1.05)
        total_years = long_paths + dt
        np.testing.assert_allclose(liquidation["Mean CAGR"], np.exp(total_log / total_years) - 1)
        self.assertLess(abs(liquidation["Mean CAGR"] - (-0.04)), 0.01)
        np.testing.assert_allclose(liquidation["Median CAGR"], -0.04)

    def test_summary_converts_momentum_days_into_years_in_each_bucket(self):
        # Columns are VOO, VXUS, and SGOV. Each row is one path.
        # Path 0 holds VOO for three quarters and SGOV for one. Path 1 stays in SGOV.
        days = np.array([
            [3, 0, 1],
            [0, 0, 4],
        ])

        summary = summarize_momentum_time(days, ["VOO", "VXUS", "SGOV"], dt=0.25)

        self.assertEqual(summary["Bucket"].tolist(), ["VOO", "VXUS", "SGOV"])
        np.testing.assert_allclose(summary["Mean Years"], [0.375, 0.0, 0.625])
        np.testing.assert_allclose(summary["Median Years"], [0.375, 0.0, 0.625])
        np.testing.assert_allclose(summary["Share of Time"], [0.375, 0.0, 0.625])

    def test_summary_limits_momentum_time_to_the_sleeve(self):
        days = np.array([[2, 5, 1]])

        summary = summarize_momentum_time(
            days,
            ["VOO", "DBMF", "SGOV"],
            dt=0.5,
            include=np.array([True, False, True]),
        )

        self.assertEqual(summary["Bucket"].tolist(), ["VOO", "SGOV"])
        np.testing.assert_allclose(summary["Mean Years"], [1.0, 0.5])
        np.testing.assert_allclose(summary["Share of Time"], [1.0 / 4.0, 0.5 / 4.0])

    def test_summary_collapses_a_bucket_held_together_into_one_row(self):
        # XLK and XLY were held together for four days. XLE was not held. SGOV got the other day.
        days = np.array([[4, 4, 0, 1]])
        labels = ["XLK", "XLY", "XLE", "SGOV"]
        signals = resolve_asset_signals(
            labels,
            sma_exit=[1, 1, 1, 1],
            sma_enter=[1, 1, 1, 1],
            asset_type=["Momentum", "Momentum", "Momentum", "Momentum"],
            bucket=["Cyclical", "Cyclical", None, None],
        )
        book = SignalBook(signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0, cash_index=3)

        collapsed, names = candidate_momentum_days(days, book, labels)
        summary = summarize_momentum_time(collapsed, names, dt=0.25)

        self.assertEqual(names, ["Cyclical", "XLE", "SGOV"])
        np.testing.assert_allclose(collapsed[0], [4, 0, 1])
        np.testing.assert_allclose(summary["Share of Time"], [0.8, 0.0, 0.2])

    def test_momentum_cagr_annualizes_only_the_days_spent_in_that_bucket(self):
        # Path 0 holds VOO for a year and compounds 10%. Path 1 holds VOO for a year
        # and compounds 21%, and holds SGOV for a year that loses 10%. VXUS is never held.
        days = np.array([
            [1, 0, 0],
            [1, 0, 1],
        ])
        log_growth = np.array([
            [np.log(1.10), 0.0, 0.0],
            [np.log(1.21), 0.0, np.log(0.90)],
        ])

        summary = summarize_momentum_time(
            days, ["VOO", "VXUS", "SGOV"], dt=1.0, log_growth=log_growth
        )

        voo = summary.loc[summary["Bucket"] == "VOO"].iloc[0]
        vxus = summary.loc[summary["Bucket"] == "VXUS"].iloc[0]
        sgov = summary.loc[summary["Bucket"] == "SGOV"].iloc[0]
        np.testing.assert_allclose(voo["Mean CAGR"], np.sqrt(1.10 * 1.21) - 1)
        np.testing.assert_allclose(voo["Median CAGR"], 0.155)
        self.assertTrue(np.isnan(vxus["Mean CAGR"]))
        self.assertTrue(np.isnan(vxus["Median CAGR"]))
        np.testing.assert_allclose(sgov["Mean CAGR"], -0.10)
        np.testing.assert_allclose(sgov["Median CAGR"], -0.10)

    def test_one_day_in_a_bucket_cannot_inflate_the_mean_cagr(self):
        # Forty paths lose 4% over a full year in SGOV. One path gains 5% on a single day.
        dt = 1 / 252
        long_paths = 40
        days = np.zeros((long_paths + 1, 1))
        days[:long_paths, 0] = 252
        days[long_paths, 0] = 1
        log_growth = np.zeros_like(days)
        log_growth[:long_paths, 0] = np.log(0.96)
        log_growth[long_paths, 0] = np.log(1.05)

        summary = summarize_momentum_time(days, ["SGOV"], dt=dt, log_growth=log_growth)

        total_log = long_paths * np.log(0.96) + np.log(1.05)
        total_years = long_paths + dt
        np.testing.assert_allclose(summary["Mean CAGR"].iloc[0], np.exp(total_log / total_years) - 1)
        self.assertLess(abs(summary["Mean CAGR"].iloc[0] - (-0.04)), 0.01)

    def test_bucket_cagr_uses_the_return_earned_while_its_members_were_held(self):
        days = np.array([[4, 4, 0, 1]])
        log_growth = np.array([[np.log(1.2), np.log(1.2), 0.0, np.log(0.95)]])
        labels = ["XLK", "XLY", "XLE", "SGOV"]
        signals = resolve_asset_signals(
            labels,
            sma_exit=[1, 1, 1, 1],
            sma_enter=[1, 1, 1, 1],
            asset_type=["Momentum", "Momentum", "Momentum", "Momentum"],
            bucket=["Cyclical", "Cyclical", None, None],
        )
        book = SignalBook(signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0, cash_index=3)

        collapsed_log = candidate_momentum_log(log_growth, book)
        collapsed, names = candidate_momentum_days(days, book, labels)
        summary = summarize_momentum_time(collapsed, names, dt=0.25, log_growth=collapsed_log)

        cyclical = summary.loc[summary["Bucket"] == "Cyclical"].iloc[0]
        np.testing.assert_allclose(cyclical["Mean CAGR"], 1.2 ** (1 / (4 * 0.25)) - 1)


class MomentumTimeTest(unittest.TestCase):
    def test_sleeve_time_follows_trades_without_a_daily_tally(self):
        # Flat prices keep every gap at zero. A negative cash hurdle lets both equity
        # names clear it, so each takes a third and the open third stays in SGOV.
        # The book is opened in cash on purpose. An unstarted book would score the
        # warmup path and might leave cash before day one. Trade days are 2 and 4
        # of a five-day run. The first two days are that opening cash book.
        labels = ["VOO", "VXUS", "SGOV"]
        signals = resolve_asset_signals(
            labels,
            sma_exit=[1, 1, 1],
            sma_enter=[1, 1, 1],
            asset_type=["Momentum", "Momentum", "Momentum"],
        )
        book = SignalBook(
            signals,
            num_paths=2,
            cash_drift=-1.0,
            cash_volatility=0.0,
            cash_index=2,
            trade_interval=2,
        )
        book.start(in_cash=True)
        days = np.zeros((2, 3), dtype=np.int64)

        simulate_with_rebalance(
            S0=np.array([0.0, 0.0, 1.0]),
            drift=np.zeros(3),
            volatility=np.zeros(3),
            T=5 / 252,
            dt=1 / 252,
            num_paths=2,
            lower_threshold=np.zeros(3),
            upper_threshold=np.ones(3),
            target_allocation=np.array([0.0, 0.0, 1.0]),
            signal=book,
            momentum_days=days,
        )

        np.testing.assert_array_equal(days, [[3, 3, 5], [3, 3, 5]])

    def test_days_before_the_first_trade_stay_in_the_opening_bucket(self):
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
            cash_drift=-1.0,
            cash_volatility=0.0,
            cash_index=1,
            trade_interval=25,
        )
        book.start(in_cash=True)
        days = np.zeros((1, 2), dtype=np.int64)

        simulate_with_rebalance(
            S0=np.array([0.0, 1.0]),
            drift=np.zeros(2),
            volatility=np.zeros(2),
            T=3 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            target_allocation=np.array([0.0, 1.0]),
            signal=book,
            momentum_days=days,
        )

        np.testing.assert_array_equal(days, [[0, 3]])

    def test_a_trade_day_still_belongs_to_the_bucket_held_that_day(self):
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
            cash_drift=-1.0,
            cash_volatility=0.0,
            cash_index=1,
            trade_interval=1,
        )
        book.start(in_cash=True)
        days = np.zeros((1, 2), dtype=np.int64)

        simulate_with_rebalance(
            S0=np.array([0.0, 1.0]),
            drift=np.zeros(2),
            volatility=np.zeros(2),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            target_allocation=np.array([0.0, 1.0]),
            signal=book,
            momentum_days=days,
        )

        np.testing.assert_array_equal(days, [[0, 1]])
        self.assertEqual(book.selection[0], 0)

    def test_held_cagr_belongs_to_the_bucket_that_earned_the_day(self):
        # One day, zero volatility, 10% drift. The sleeve opens in SGOV, and that day
        # stays with SGOV even though the trade then selects VOO.
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
            cash_drift=-1.0,
            cash_volatility=0.0,
            cash_index=1,
            trade_interval=1,
        )
        book.start(in_cash=True)
        days = np.zeros((1, 2), dtype=np.int64)
        log_growth = np.zeros((1, 2))

        simulate_with_rebalance(
            S0=np.array([0.0, 1.0]),
            drift=np.full(2, 0.10),
            volatility=np.zeros(2),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            target_allocation=np.array([0.0, 1.0]),
            signal=book,
            momentum_days=days,
            momentum_log=log_growth,
        )

        np.testing.assert_allclose(log_growth[0], [0.0, 0.10 / 252])


class TotalGbmWarmupTest(unittest.TestCase):
    """A shadow GBM fills the longest SMA before the portfolio opens."""

    def test_total_gbm_compounds_each_path_from_one(self):
        growth = np.log(1.2)
        drift = np.zeros((2, 2))
        drift[0, 0] = growth
        closes = total_gbm(drift, np.zeros_like(drift), dt=1.0, num_days=4)

        self.assertEqual(closes.shape, (4, 2, 2))
        np.testing.assert_allclose(closes[:, 0, 0], [1.2, 1.2 ** 2, 1.2 ** 3, 1.2 ** 4])
        np.testing.assert_allclose(closes[:, 1, :], 1.0)
        np.testing.assert_allclose(closes[:, 0, 1], 1.0)

    def test_lead_covers_the_longer_window_and_the_closes_left_out_of_it(self):
        book = self.book(num_paths=1)

        # Enter and exit are 3, and the 2 newest closes stay out of the average.
        self.assertEqual(book.lead_days, 5)

    def test_each_path_can_open_in_cash_or_in_its_momentum_bucket(self):
        # Zero volatility makes the lead deterministic. Path 0 trends, so its
        # bucket clears the cash hurdle. Path 1 is flat and stays in SGOV.
        book = self.book(num_paths=2)
        growth = np.log(1.2)
        drift = np.zeros((2, 3))
        drift[0, :2] = growth
        days = np.zeros((2, 3), dtype=np.int64)

        prices, _lower, _upper = simulate_with_rebalance(
            S0=np.array([0.60, 0.40, 0.00]),
            drift=drift,
            volatility=np.zeros((2, 3)),
            T=1.0,
            dt=1.0,
            num_paths=2,
            lower_threshold=np.zeros(3),
            upper_threshold=np.ones(3),
            target_allocation=np.array([0.60, 0.40, 0.00]),
            initial_value=100.0,
            signal=book,
            momentum_days=days,
        )

        self.assertEqual(prices.shape[0], 2)
        np.testing.assert_allclose(prices[0, 0], [60.0, 40.0, 0.0])
        np.testing.assert_allclose(prices[0, 1], [0.0, 0.0, 100.0])
        np.testing.assert_allclose(prices[1, 0], [72.0, 48.0, 0.0])
        np.testing.assert_allclose(prices[1, 1], [0.0, 0.0, 100.0])
        np.testing.assert_array_equal(days, [[1, 1, 0], [0, 0, 1]])
        self.assertEqual(book._steps_seen, book.lead_days + 1)
        np.testing.assert_allclose(book._history[book._pos, 0, 0], 1.2 ** (book.lead_days + 1))
        self.assertFalse(book.in_cash[0])
        self.assertTrue(book.in_cash[1])

    def test_a_flat_lead_stays_in_cash_when_nothing_beats_the_hurdle(self):
        book = self.book(num_paths=1)

        prices, _lower, _upper = simulate_with_rebalance(
            S0=np.array([0.60, 0.40, 0.00]),
            drift=np.zeros(3),
            volatility=np.zeros(3),
            T=1.0,
            dt=1.0,
            num_paths=1,
            lower_threshold=np.zeros(3),
            upper_threshold=np.ones(3),
            target_allocation=np.array([0.60, 0.40, 0.00]),
            initial_value=100.0,
            signal=book,
        )

        np.testing.assert_allclose(prices[0, 0], [0.0, 0.0, 100.0])
        self.assertTrue(book.in_cash[0])

    def test_an_explicit_cash_open_is_left_alone(self):
        book = self.book(num_paths=1)
        book.start(in_cash=True)

        prices, _lower, _upper = simulate_with_rebalance(
            S0=np.array([40.0, 60.0, 0.0]),
            drift=np.full(3, np.log(1.2)),
            volatility=np.zeros(3),
            T=1.0,
            dt=1.0,
            num_paths=1,
            lower_threshold=np.zeros(3),
            upper_threshold=np.ones(3),
            target_allocation=np.array([0.60, 0.40, 0.00]),
            signal=book,
        )

        np.testing.assert_allclose(prices[0, 0], [40.0, 60.0, 0.0])
        self.assertEqual(book._steps_seen, 2)
        self.assertTrue(book.in_cash[0])

    def test_warmup_does_not_age_the_opening_regime(self):
        # Churn draws expansion. The recorded open and the first step stay in
        # churn. The second step is that draw. The lead does not take one.
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
            cash_drift=0.05,
            cash_volatility=0.0,
            cash_index=1,
            top_num=1,
        )
        cycle = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        regime_states = np.zeros((3, 1), dtype=int)

        simulate_with_rebalance(
            S0=np.array([0.0, 1.0]),
            drift=np.zeros(2),
            volatility=np.zeros(2),
            T=2.0,
            dt=1.0,
            num_paths=1,
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            target_allocation=np.array([0.0, 1.0]),
            regime_drift=np.zeros((3, 2)),
            regime_volatility=np.zeros((3, 2)),
            trigger_index=0,
            regime_states=regime_states,
            signal=book,
            transition=cycle,
            transition_interval=1,
        )

        np.testing.assert_array_equal(regime_states[:, 0], [CHURN, CHURN, EXPANSION])

    def book(self, num_paths):
        signals = resolve_asset_signals(
            ["VOO", "SPMO", "SGOV"],
            sma_exit=[3, 3, 1],
            sma_enter=[3, 3, 1],
            asset_type=["Momentum", "Momentum", "Momentum"],
            sma_exclusion=[2, 2, 0],
            bucket=["Equity", "Equity", None],
        )
        return SignalBook(
            signals,
            num_paths=num_paths,
            cash_drift=0.05,
            cash_volatility=0.0,
            cash_index=2,
            trade_interval=25,
            top_num=1,
        )


class EndingCagrTest(unittest.TestCase):
    def test_compounds_the_starting_value_into_the_ending_value(self):
        ending = 100 * (1.10 ** 10)
        np.testing.assert_allclose(ending_cagr(ending, 100, 10), 0.10)
        np.testing.assert_allclose(ending_cagr(100, 100, 10), 0.0)

    def test_missing_horizon_or_starting_value_has_no_rate(self):
        self.assertTrue(np.isnan(ending_cagr(200, 100, 0)))
        self.assertTrue(np.isnan(ending_cagr(200, 0, 10)))


class ValuationMetricsTest(unittest.TestCase):
    def test_each_ending_statistic_has_its_own_cagr(self):
        endings = np.array([100.0, 121.0, 144.0, 169.0, 196.0])
        paths = np.vstack([np.full(5, 100.0), endings])

        metrics = summarize_valuation_metrics(paths, years=2).set_index("Metric")

        expected = {
            "Upside 90th Percentile": np.percentile(endings, 90),
            "Expected Mean Ending Value": endings.mean(),
            "Median Ending Value": np.median(endings),
            "95% Confidence Portfolio Floor (5th%)": np.percentile(endings, 5),
        }
        for name, ending in expected.items():
            np.testing.assert_allclose(metrics.loc[name, "Value"], ending)
            np.testing.assert_allclose(metrics.loc[name, "CAGR"], ending_cagr(ending, 100, 2))
        np.testing.assert_allclose(metrics.loc["Initial Portfolio Value", "Value"], 100)
        self.assertTrue(np.isnan(metrics.loc["Initial Portfolio Value", "CAGR"]))

    def test_each_row_gets_the_same_rank_of_path_drawdowns(self):
        # Three books. The first gives back a fifth of a new high and recovers.
        # The second never makes a high and is cut in half. The third holds a
        # modest high, then is sold down to 40. Drawdown is that loss, kept
        # after any later recovery.
        paths = np.array([
            [100.0, 100.0, 100.0],
            [150.0, 80.0, 110.0],
            [120.0, 50.0, 110.0],
            [180.0, 90.0, 40.0],
        ])
        drawdowns = np.array([-0.20, -0.50, 40.0 / 110.0 - 1.0])

        metrics = summarize_valuation_metrics(paths, years=3).set_index("Metric")

        np.testing.assert_allclose(path_max_drawdown(paths), drawdowns)
        np.testing.assert_allclose(
            metrics.loc["Upside 90th Percentile", "Max Drawdown"], np.percentile(drawdowns, 90)
        )
        np.testing.assert_allclose(
            metrics.loc["Expected Mean Ending Value", "Max Drawdown"], drawdowns.mean()
        )
        np.testing.assert_allclose(
            metrics.loc["Median Ending Value", "Max Drawdown"], np.median(drawdowns)
        )
        np.testing.assert_allclose(
            metrics.loc["95% Confidence Portfolio Floor (5th%)", "Max Drawdown"],
            np.percentile(drawdowns, 5),
        )
        self.assertTrue(np.isnan(metrics.loc["Initial Portfolio Value", "Max Drawdown"]))
        self.assertLess(
            metrics.loc["95% Confidence Portfolio Floor (5th%)", "Max Drawdown"],
            metrics.loc["Upside 90th Percentile", "Max Drawdown"],
        )

    def test_a_book_that_never_falls_or_never_trades_has_no_loss(self):
        rising = np.array([[100.0], [110.0], [110.0]])
        ruined = np.array([[100.0, 0.0], [0.0, 0.0]])

        np.testing.assert_allclose(path_max_drawdown(rising), [0.0])
        np.testing.assert_allclose(path_max_drawdown(ruined), [-1.0, 0.0])

    def test_sharpe_is_annualized_excess_return_per_unit_of_variation(self):
        # Two year-long steps. Cash pays 5% simple, the continuous drift log(1.05).
        # The first book earns 10% and then nothing, so its excess averages to zero.
        # The second earns 20% and then 10%.
        paths = np.array([
            [100.0, 100.0],
            [110.0, 120.0],
            [110.0, 132.0],
        ])
        risk_free = np.log(1.05)
        returns = paths[1:] / paths[:-1] - 1.0
        excess = returns - (np.exp(risk_free) - 1.0)
        expected = excess.mean(axis=0) / excess.std(axis=0, ddof=1)

        ratios = path_sharpe_ratio(paths, years=2, risk_free=risk_free)

        np.testing.assert_allclose(ratios, expected)
        np.testing.assert_allclose(ratios[0], 0.0)
        # The same steps packed into a quarter of the horizon are twice the annual ratio.
        compressed = path_sharpe_ratio(paths, years=0.5, risk_free=0.0)
        uncompressed = path_sharpe_ratio(paths, years=2, risk_free=0.0)
        np.testing.assert_allclose(compressed, uncompressed * 2.0)

        metrics = summarize_valuation_metrics(paths, years=2, risk_free=risk_free).set_index("Metric")
        np.testing.assert_allclose(
            metrics.loc["Upside 90th Percentile", "Sharpe Ratio"], np.percentile(expected, 90)
        )
        np.testing.assert_allclose(
            metrics.loc["Expected Mean Ending Value", "Sharpe Ratio"], expected.mean()
        )
        np.testing.assert_allclose(
            metrics.loc["Median Ending Value", "Sharpe Ratio"], np.median(expected)
        )
        np.testing.assert_allclose(
            metrics.loc["95% Confidence Portfolio Floor (5th%)", "Sharpe Ratio"],
            np.percentile(expected, 5),
        )
        self.assertTrue(np.isnan(metrics.loc["Initial Portfolio Value", "Sharpe Ratio"]))
        self.assertLess(
            metrics.loc["95% Confidence Portfolio Floor (5th%)", "Sharpe Ratio"],
            metrics.loc["Upside 90th Percentile", "Sharpe Ratio"],
        )

    def test_a_path_without_variation_or_a_second_return_has_no_sharpe(self):
        flat = np.array([[100.0], [110.0], [121.0]])
        varying = np.array([[100.0], [120.0], [108.0]])
        one_step = np.array([[100.0, 100.0], [110.0, 90.0]])
        ruined = np.array([[100.0], [0.0], [0.0]])

        self.assertTrue(np.isnan(path_sharpe_ratio(flat, years=2, risk_free=np.log(1.10))).all())
        self.assertTrue(np.isnan(path_sharpe_ratio(one_step, years=1, risk_free=0.0)).all())
        self.assertTrue(np.isnan(path_sharpe_ratio(varying, years=0, risk_free=0.0)).all())
        # The step that reaches zero is a total loss. The following step has no
        # prior value, so it contributes no further change.
        loss = np.array([-1.0, 0.0])
        np.testing.assert_allclose(
            path_sharpe_ratio(ruined, years=2, risk_free=0.0),
            [loss.mean() / loss.std(ddof=1)],
        )

        combined = np.concatenate([varying, flat], axis=1)
        ratios = path_sharpe_ratio(combined, years=2, risk_free=0.0)
        metrics = summarize_valuation_metrics(combined, years=2).set_index("Metric")
        np.testing.assert_allclose(
            metrics.loc["Expected Mean Ending Value", "Sharpe Ratio"],
            ratios[np.isfinite(ratios)].mean(),
        )


class AllAlternateBookTest(unittest.TestCase):
    """Asset Parameters is an all-alternate book. Bands snap back to those targets.

    With no momentum sleeve, the tactical rule would park the book in SGOV until
    an SMA window fills, then hand whatever is left to the name furthest above
    its average. An alternate keeps the weight on the sheet instead.
    """

    WORKBOOK = r"C:\Users\Evan\VirtualBox VMs\Sandbox\PortfolioManagement\portfolio_gbm_calculator.xlsx"
    SHEET = "Asset Parameters"

    def sheet(self):
        frame = pd.read_excel(self.WORKBOOK, sheet_name=self.SHEET)
        labels = [str(ticker).strip() for ticker in frame["Ticker"]]
        kinds = frame["Asset Type"].astype(str).str.strip().str.lower()
        self.assertTrue(kinds.isin(["alternate", "alternative"]).all())
        self.assertEqual(labels, ["VOO", "VXUS", "SGOV", "DBMF", "BTC"])
        return frame, labels

    def book(self, frame, labels, num_paths=1, cash_drift=0.40):
        signals = resolve_asset_signals(
            labels,
            frame["SMA Exit"],
            frame["SMA Enter"],
            asset_type=frame["Asset Type"],
            sma_exclusion=frame["SMA-exclusion"],
        )
        return SignalBook(
            signals,
            num_paths=num_paths,
            cash_drift=cash_drift,
            cash_volatility=0.0,
            cash_index=labels.index("SGOV"),
        )

    def test_sheet_targets_stay_put_before_any_average_exists(self):
        frame, labels = self.sheet()
        targets = frame["Target Allocation"].to_numpy(dtype=float)
        book = self.book(frame, labels, num_paths=2)

        weights = book.target_weights(targets)

        np.testing.assert_allclose(weights, np.broadcast_to(targets, (2, len(labels))))
        self.assertTrue(book.held().all())
        drift, volatility = book.apply(
            np.array([[0.22, 0.13, 0.04, 0.04, 0.55]] * 2),
            np.array([[0.14, 0.13, 0.0, 0.10, 0.50]] * 2),
        )
        np.testing.assert_allclose(drift, [[0.22, 0.13, 0.04, 0.04, 0.55]] * 2)
        np.testing.assert_allclose(volatility, [[0.14, 0.13, 0.0, 0.10, 0.50]] * 2)

    def test_a_band_breach_snaps_back_to_the_sheet_targets(self):
        frame, labels = self.sheet()
        targets = frame["Target Allocation"].to_numpy(dtype=float)
        book = self.book(frame, labels)
        # VOO is above 0.80 and DBMF is below 0.15. VXUS and SGOV stay at zero.
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([85.0, 0.0, 0.0, 10.0, 5.0]),
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=frame["Lower Band"].to_numpy(dtype=float),
            upper_threshold=frame["Upper Band"].to_numpy(dtype=float),
            target_allocation=targets,
            signal=book,
            rebalance_interval=1,
        )

        np.testing.assert_allclose(prices[1, 0], targets * 100.0)
        np.testing.assert_array_equal(upper_counts, [[1, 0, 0, 0, 0]])
        np.testing.assert_array_equal(lower_counts, [[0, 0, 0, 1, 0]])

    def test_a_book_inside_the_bands_keeps_each_alternate_return(self):
        frame, labels = self.sheet()
        targets = frame["Target Allocation"].to_numpy(dtype=float)
        book = self.book(frame, labels, cash_drift=0.40)
        start = targets * 100.0
        drift = np.array([0.0, 0.0, 0.0, 0.0, 0.50])
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=start,
            drift=drift,
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=frame["Lower Band"].to_numpy(dtype=float),
            upper_threshold=frame["Upper Band"].to_numpy(dtype=float),
            target_allocation=targets,
            signal=book,
            rebalance_interval=1,
        )

        expected = start * np.exp(drift / 252)
        np.testing.assert_allclose(prices[1, 0], expected)
        np.testing.assert_array_equal(lower_counts + upper_counts, np.zeros((1, 5)))

    def test_a_zero_target_alternate_does_not_force_a_rebalance(self):
        labels = ["VOO", "VXUS", "SGOV", "DBMF", "BTC"]
        signals = resolve_asset_signals(
            labels,
            sma_exit=[3, 3, 3, 3, 3],
            sma_enter=[3, 3, 3, 3, 3],
            asset_type=["Alternate", "Alternate", "Alternate", "Alternate", "Alternate"],
        )
        book = SignalBook(signals, num_paths=1, cash_drift=0.40, cash_volatility=0.0, cash_index=2)
        targets = np.array([0.70, 0.0, 0.0, 0.20, 0.10])
        # VXUS is below a lower band of 0.50, but its target is zero, so it cannot trade.
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=targets * 100.0,
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.array([0.60, 0.50, -0.01, 0.15, 0.02]),
            upper_threshold=np.array([0.80, 1.01, 1.01, 0.27, 0.15]),
            target_allocation=targets,
            signal=book,
            rebalance_interval=1,
        )

        np.testing.assert_allclose(prices[1, 0], targets * 100.0)
        np.testing.assert_array_equal(lower_counts + upper_counts, np.zeros((1, 5)))

    def test_an_untyped_book_still_rotates_to_the_name_above_its_average(self):
        labels = ["VOO", "DBMF", "SGOV"]
        targets = np.array([0.70, 0.30, 0.0])
        signals = resolve_asset_signals(
            labels,
            sma_exit=[3, 3, 3],
            sma_enter=[3, 3, 3],
        )
        book = SignalBook(signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0, cash_index=2)
        book.load_history(np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ]))

        weights = book.target_weights(targets)

        np.testing.assert_allclose(weights[0], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
