import unittest
from unittest import mock

import numpy as np
import pandas as pd

from gbm.bt import (
    BACKTEST_YEARS,
    WARMUP_YEARS,
    backtest_bounds,
    backtest_strategy,
    load_model,
    market_backtest,
    parse_args,
    parse_start_date,
    prices_for_labels,
    slice_history,
)
from gbm.signals import SignalBook, resolve_asset_signals


class BacktestWindowTest(unittest.TestCase):
    def test_the_default_window_is_ten_years_with_one_year_of_warmup(self):
        warmup, start, end = backtest_bounds(as_of="10/22/2026")

        self.assertEqual(BACKTEST_YEARS, 10)
        self.assertEqual(WARMUP_YEARS, 1)
        self.assertEqual(start, pd.Timestamp("2016-10-22"))
        self.assertEqual(warmup, pd.Timestamp("2015-10-22"))
        self.assertEqual(end, pd.Timestamp("2026-10-22"))

    def test_start_date_sets_the_first_scored_day_and_warms_the_year_before(self):
        warmup, start, end = backtest_bounds("01/02/2018", as_of="10/22/2026")

        self.assertEqual(parse_start_date("01/02/2018"), pd.Timestamp("2018-01-02"))
        self.assertEqual(start, pd.Timestamp("2018-01-02"))
        self.assertEqual(warmup, pd.Timestamp("2017-01-02"))
        self.assertEqual(end, pd.Timestamp("2026-10-22"))

    def test_a_start_date_has_to_be_mm_dd_yyyy(self):
        with self.assertRaises(ValueError):
            parse_start_date("2018-01-02")
        with self.assertRaises(ValueError):
            parse_start_date("13/40/2018")

    def test_a_start_after_the_sample_has_no_backtest(self):
        with self.assertRaises(ValueError):
            backtest_bounds("01/01/2027", as_of="10/22/2026")


class BacktestHistoryTest(unittest.TestCase):
    def test_a_time_zone_keeps_the_session_date(self):
        index = pd.DatetimeIndex([
            pd.Timestamp("2016-01-04 16:00", tz="America/New_York"),
            pd.Timestamp("2016-01-05 16:00", tz="America/New_York"),
        ])
        frame = pd.DataFrame({"VOO": [1.0, 2.0]}, index=index)

        aligned = prices_for_labels(frame, ["VOO"])

        self.assertEqual(list(aligned.index), [pd.Timestamp("2016-01-04"), pd.Timestamp("2016-01-05")])

    def test_a_sleeve_label_uses_the_market_column(self):
        frame = pd.DataFrame(
            {"VOO": [1.0, 2.0], "SGOV": [1.0, 1.0]},
            index=pd.to_datetime(["2016-01-04", "2016-01-05"]),
        )

        aligned = prices_for_labels(frame, ["VOO/VIIX", "SGOV"])

        self.assertEqual(list(aligned.columns), ["VOO/VIIX", "SGOV"])
        np.testing.assert_allclose(aligned["VOO/VIIX"], [1.0, 2.0])

    def test_warmup_keeps_the_full_year_when_it_covers_the_signal(self):
        index = pd.bdate_range("2016-01-01", periods=30)
        prices = pd.DataFrame({"A": np.arange(30, dtype=float) + 1.0}, index=index)

        warmup, scored = slice_history(prices, index[20], index[0], lead_days=5)

        self.assertEqual(len(warmup), 20)
        self.assertEqual(warmup.index[0], index[0])
        self.assertEqual(warmup.index[-1], index[19])
        self.assertEqual(scored.index[0], index[20])

    def test_warmup_reaches_back_when_the_year_is_shorter_than_the_signal(self):
        index = pd.bdate_range("2016-01-01", periods=30)
        prices = pd.DataFrame({"A": np.arange(30, dtype=float) + 1.0}, index=index)

        warmup, scored = slice_history(prices, index[20], index[18], lead_days=5)

        self.assertEqual(list(warmup.index), list(index[15:20]))
        self.assertEqual(scored.index[0], index[20])

    def test_a_short_history_cannot_fill_the_signal(self):
        index = pd.bdate_range("2016-01-01", periods=4)
        prices = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=index)

        with self.assertRaises(ValueError):
            slice_history(prices, index[2], index[0], lead_days=10)


class BacktestReplayTest(unittest.TestCase):
    def test_a_band_breach_snaps_the_historical_book(self):
        # The supplied close doubles the first asset. That weight leaves the
        # band, and the snap uses the grown value, not a random shock.
        valuations, lower_counts, upper_counts = backtest_strategy(
            growth=np.array([[2.0, 1.0]]),
            signal=None,
            target_allocation=np.array([0.50, 0.50]),
            lower_threshold=np.array([0.40, 0.20]),
            upper_threshold=np.array([0.60, 0.80]),
            initial_value=100.0,
            rebalance_interval=1,
        )

        np.testing.assert_allclose(valuations[0, 0], [50.0, 50.0])
        np.testing.assert_allclose(valuations[1, 0], [75.0, 75.0])
        np.testing.assert_array_equal(upper_counts, [[1, 0]])
        np.testing.assert_array_equal(lower_counts, [[0, 0]])

    def test_the_monthly_check_waits_the_same_twenty_one_days(self):
        # The first asset leaves its band during the month. The book keeps
        # that drift until the 21st session, then snaps back to the targets.
        growth = np.tile([1.03, 1.0], (21, 1))
        valuations, _lower, upper_counts = backtest_strategy(
            growth=growth,
            signal=None,
            target_allocation=np.array([0.60, 0.40]),
            lower_threshold=np.array([0.50, 0.20]),
            upper_threshold=np.array([0.70, 0.50]),
            initial_value=100.0,
            rebalance_interval=21,
        )

        drifted = np.array([60.0, 40.0]) * np.array([1.03 ** 20, 1.0])
        self.assertGreater(drifted[0] / drifted.sum(), 0.70)
        np.testing.assert_allclose(valuations[20, 0], drifted)
        snapped = drifted * np.array([1.03, 1.0])
        np.testing.assert_allclose(valuations[21, 0], snapped.sum() * np.array([0.60, 0.40]))
        np.testing.assert_array_equal(upper_counts, [[1, 0]])

    def test_warmup_gains_are_not_part_of_the_scored_book(self):
        # VOO trends through the warmup and is the sleeve on the open.
        # The open is still the starting dollars. The next scored day earns
        # VOO's own close, and the flat warmup year is not a profit.
        book = self.momentum_book(beat_cash=False, cash_drift=0.0)
        warmup = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
        ])
        days = np.zeros((1, 3), dtype=np.int64)

        def boom(*_args, **_kwargs):
            raise AssertionError("backtest drew a random shock")

        with mock.patch("numpy.random.standard_normal", boom):
            valuations, _lower, _upper = backtest_strategy(
                growth=np.array([
                    [1.0, 1.0, 1.0],
                    [1.1, 1.0, 1.0],
                ]),
                signal=book,
                target_allocation=np.array([0.60, 0.40, 0.00]),
                lower_threshold=np.zeros(3),
                upper_threshold=np.ones(3),
                initial_value=100.0,
                warmup_closes=warmup,
                momentum_days=days,
            )

        np.testing.assert_allclose(valuations[0, 0], [100.0, 0.0, 0.0])
        np.testing.assert_allclose(valuations[2, 0], [110.0, 0.0, 0.0])
        np.testing.assert_array_equal(days, [[2, 0, 0]])
        self.assertFalse(book.in_cash[0])

    def test_a_flat_warmup_opens_in_cash_and_earns_the_cash_close(self):
        book = self.momentum_book(beat_cash=True, cash_drift=0.05)
        warmup = np.ones((4, 3))

        valuations, _lower, _upper = backtest_strategy(
            growth=np.array([[1.1, 1.0, 1.01]]),
            signal=book,
            target_allocation=np.array([0.60, 0.40, 0.00]),
            lower_threshold=np.zeros(3),
            upper_threshold=np.ones(3),
            initial_value=100.0,
            warmup_closes=warmup,
        )

        np.testing.assert_allclose(valuations[1, 0], [0.0, 0.0, 101.0])
        self.assertTrue(book.in_cash[0])

    def test_an_unheld_sleeve_earns_the_cash_close(self):
        # The risky close finishes the warmup below its average, so the book
        # keeps the dollars on that row but credits cash's historical return.
        labels = ["VOO", "SGOV"]
        signals = resolve_asset_signals(labels, sma_exit=[2, 2], sma_enter=[2, 2])
        book = SignalBook(signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0, cash_index=1)
        warmup = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [0.5, 1.0],
        ])

        valuations, _lower, _upper = backtest_strategy(
            growth=np.array([[2.0, 1.1]]),
            signal=book,
            target_allocation=np.array([0.60, 0.40]),
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            initial_value=100.0,
            warmup_closes=warmup,
            rebalance_interval=21,
        )

        np.testing.assert_allclose(valuations[0, 0], [60.0, 40.0])
        np.testing.assert_allclose(valuations[1, 0], [66.0, 44.0])
        np.testing.assert_allclose(book._history[book._pos, 0], [1.0, 1.1])

    def momentum_book(self, beat_cash, cash_drift):
        signals = resolve_asset_signals(
            ["VOO", "SPMO", "SGOV"],
            sma_exit=[3, 3, 1],
            sma_enter=[3, 3, 1],
            asset_type=["Momentum", "Momentum", "Momentum"],
        )
        return SignalBook(
            signals,
            num_paths=1,
            cash_drift=cash_drift,
            cash_volatility=0.0,
            cash_index=2,
            trade_interval=25,
            top_num=1,
            beat_cash=beat_cash,
        )


class MarketBacktestTest(unittest.TestCase):
    def test_the_scored_path_starts_on_the_requested_date(self):
        index = pd.bdate_range("2014-01-01", "2016-12-30")
        frame = pd.DataFrame({"VOO": 1.0, "SGOV": 1.0}, index=index)

        result = market_backtest(
            frame,
            labels=["VOO", "SGOV"],
            signal=None,
            target_allocation=np.array([0.50, 0.50]),
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            start_date="01/04/2016",
            as_of="12/30/2016",
            initial_value=100.0,
        )

        self.assertEqual(result.eval_start, pd.Timestamp("2016-01-04"))
        self.assertLess(result.dates[0], result.eval_start)
        self.assertGreaterEqual(result.dates[0], pd.Timestamp("2015-01-04"))
        self.assertGreaterEqual(result.dates[1], result.eval_start)
        self.assertEqual(result.dates[-1], index[index >= "2016-01-04"][-1])
        np.testing.assert_allclose(result.valuations.sum(axis=2), 100.0)

    def test_the_default_sample_is_the_last_ten_years_of_closes(self):
        index = pd.bdate_range("2010-01-01", "2026-10-22")
        frame = pd.DataFrame({"VOO": 1.0, "SGOV": 1.0}, index=index)

        result = market_backtest(
            frame,
            labels=["VOO", "SGOV"],
            signal=None,
            target_allocation=np.array([0.50, 0.50]),
            lower_threshold=np.zeros(2),
            upper_threshold=np.ones(2),
            as_of="10/22/2026",
            initial_value=100.0,
        )

        self.assertEqual(result.eval_start, pd.Timestamp("2016-10-22"))
        self.assertEqual(result.warmup_start, pd.Timestamp("2015-10-22"))
        self.assertLess(result.dates[0], result.eval_start)
        self.assertGreaterEqual(result.dates[0], result.warmup_start)
        self.assertEqual(result.dates[-1], pd.Timestamp("2026-10-22"))

    def test_a_ticker_that_lists_after_the_start_is_named(self):
        index = pd.bdate_range("2015-01-01", "2017-12-29")
        frame = pd.DataFrame({"VOO": 1.0, "SGOV": np.nan}, index=index)
        frame.loc["2017-06-01":, "SGOV"] = 1.0

        with self.assertRaises(ValueError) as caught:
            market_backtest(
                frame,
                labels=["VOO", "SGOV"],
                signal=None,
                target_allocation=np.array([0.50, 0.50]),
                lower_threshold=np.zeros(2),
                upper_threshold=np.ones(2),
                start_date="01/04/2016",
                as_of="12/29/2017",
                initial_value=100.0,
            )

        message = str(caught.exception)
        self.assertIn("SGOV", message)
        self.assertIn("01/04/2016", message)
        self.assertIn("--start_date", message)


class BacktestCommandTest(unittest.TestCase):
    def test_sheet_name_and_start_date_are_flags(self):
        default = parse_args([])
        chosen = parse_args(["--start_date", "03/01/2014", "--sheet_name", "Sector Momentum Playbook"])
        hyphenated = parse_args([
            "--sheet-name",
            "Sector Momentum Playbook",
            "--start-date",
            "03/01/2014",
            "--top_num",
            "2",
            "--no-beat_cash",
            "--dynamic-rebalance",
        ])

        self.assertEqual(default.sheet_name, "Asset Parameters")
        self.assertIsNone(default.start_date)
        self.assertFalse(default.dynamic_rebalance)
        self.assertEqual(chosen.start_date, "03/01/2014")
        self.assertEqual(chosen.sheet_name, "Sector Momentum Playbook")
        self.assertEqual(hyphenated.sheet_name, "Sector Momentum Playbook")
        self.assertEqual(hyphenated.start_date, "03/01/2014")
        self.assertEqual(hyphenated.top_num, 2)
        self.assertFalse(hyphenated.beat_cash)
        self.assertTrue(hyphenated.dynamic_rebalance)

    def test_sheet_name_selects_the_model(self):
        core = load_model("Asset Parameters")
        sectors = load_model("Sector Momentum Playbook", top_num=2, beat_cash=False)

        self.assertIn("SGOV", core.labels)
        self.assertIn("SGOV", sectors.labels)
        self.assertNotEqual(core.labels, sectors.labels)
        self.assertTrue(core.signal.uses_momentum_sleeve)
        self.assertFalse(core.signal.static_allocation)
        self.assertTrue(sectors.signal.uses_momentum_sleeve)
        self.assertEqual(sectors.signal.top_num, 2)
        self.assertFalse(sectors.signal.beat_cash)
        self.assertEqual(core.signal.top_num, 3)


if __name__ == "__main__":
    unittest.main()
