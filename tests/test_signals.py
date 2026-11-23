import unittest

import numpy as np
import pandas as pd

from gbm.gbm_step import dynamic_rebalance, momentum_shift, monthly_momentum, simulate_with_rebalance
from gbm.signals import SignalBook, current_momentum_bucket, resolve_asset_signals, tactical_weights


class SignalResolutionTest(unittest.TestCase):
    def test_tracked_index_outranks_sma_windows_and_none_is_never_liquidated(self):
        signals = resolve_asset_signals(
            ["VOO/VIIX", "SPMO", "BTC", "CSVYX", "GLD"],
            sma_exit=[200, 50, -1, 100, -1],
            sma_enter=[200, 20, -1, 50, -1],
            tracked_index=[None, "VOO", "none", "", "non"],
        )

        self.assertEqual(signals[0].source, 0)
        self.assertEqual((signals[0].exit, signals[0].enter), (200, 200))
        self.assertFalse(signals[0].follows_index)
        self.assertFalse(signals[0].never_liquidate)

        self.assertTrue(signals[1].follows_index)
        self.assertEqual(signals[1].source, 0)
        self.assertEqual((signals[1].exit, signals[1].enter), (200, 200))
        self.assertFalse(signals[1].never_liquidate)

        self.assertTrue(signals[2].never_liquidate)
        self.assertFalse(signals[2].follows_index)
        self.assertTrue(signals[4].never_liquidate)

        self.assertFalse(signals[3].follows_index)
        self.assertFalse(signals[3].never_liquidate)
        self.assertEqual((signals[3].exit, signals[3].enter), (100, 50))
        self.assertEqual(signals[0].exclusion, 0)

    def test_opening_exclusion_sum_stays_path_by_asset_when_the_counts_differ(self):
        signals = resolve_asset_signals(
            ["VOO", "VXUS", "SGOV"],
            sma_exit=[3, 3, 3],
            sma_enter=[3, 3, 3],
            sma_exclusion=[2, 0, 1],
        )

        book = SignalBook(signals, num_paths=4, cash_drift=0.0, cash_volatility=0.0)

        self.assertEqual(book._recent_sum.shape, (4, 3))
        np.testing.assert_allclose(book._recent_sum, np.array([[1.0, 0.0, 1.0]] * 4))

    def test_sma_exclusion_sets_the_lag_and_a_follower_uses_the_index_lag(self):
        signals = resolve_asset_signals(
            ["VOO", "SPMO", "DBMF"],
            sma_exit=[250, 50, 200],
            sma_enter=[210, 20, 200],
            tracked_index=[None, "VOO", None],
            sma_exclusion=[25, 5, None],
        )

        self.assertEqual(signals[0].exclusion, 25)
        self.assertEqual(signals[1].exclusion, 25)
        self.assertEqual((signals[1].exit, signals[1].enter), (250, 210))
        self.assertEqual(signals[2].exclusion, 0)

    def test_missing_sma_flags_default_to_a_200_day_enter_and_exit(self):
        signals = resolve_asset_signals(["DBMF"], sma_exit=[-1], sma_enter=[-1], tracked_index=[None])

        self.assertEqual((signals[0].exit, signals[0].enter), (200, 200))
        self.assertFalse(signals[0].never_liquidate)


class SignalBookTest(unittest.TestCase):
    def test_a_break_of_the_exit_average_parks_the_sleeve_in_cash_until_the_enter_average(self):
        signals = resolve_asset_signals(["VOO"], sma_exit=[3], sma_enter=[3], tracked_index=[None])
        book = SignalBook(signals, num_paths=1, cash_drift=0.05, cash_volatility=0.0)
        book.observe(np.array([[1.0]]))
        book.observe(np.array([[1.0]]))
        book.observe(np.array([[0.5]]))

        self.assertFalse(book.held()[0, 0])
        drift, volatility = book.apply(np.array([[0.20]]), np.array([[0.11]]))
        self.assertAlmostEqual(drift[0, 0], 0.05)
        self.assertAlmostEqual(volatility[0, 0], 0.0)

        book.observe(np.array([[2.0]]))

        self.assertTrue(book.held()[0, 0])
        drift, _volatility = book.apply(np.array([[0.20]]), np.array([[0.11]]))
        self.assertAlmostEqual(drift[0, 0], 0.20)

    def test_none_stays_invested_and_a_follower_uses_the_index(self):
        signals = resolve_asset_signals(
            ["VOO", "SPMO", "BTC"],
            sma_exit=[3, 3, 3],
            sma_enter=[3, 3, 3],
            tracked_index=[None, "VOO", "none"],
        )
        book = SignalBook(signals, num_paths=1, cash_drift=0.05, cash_volatility=0.0)
        flat = np.ones((1, 3))
        book.observe(flat)
        book.observe(flat)
        book.observe(np.array([[0.5, 2.0, 0.5]]))

        held = book.held()
        self.assertFalse(held[0, 0])
        self.assertFalse(held[0, 1])
        self.assertTrue(held[0, 2])

        drift, _volatility = book.apply(np.array([[0.15, 0.22, 0.55]]), np.zeros((1, 3)))
        np.testing.assert_allclose(drift[0], [0.05, 0.05, 0.55])

    def test_rotating_sleeve_is_the_group_furthest_above_its_enter_average(self):
        labels = ["VOO", "SPMO", "CSVYX", "DWAS", "DBMF", "BTC", "AVES", "TQQQ", "KMLM", "GLD", "SGOV"]
        targets = np.array([0.25, 0.00, 0.15, 0.00, 0.20, 0.10, 0.15, 0.05, 0.05, 0.05, 0.00])
        signals = resolve_asset_signals(
            labels,
            sma_exit=[250, None, 125, 125, None, None, 250, None, None, None, None],
            sma_enter=[50, None, 50, 50, None, None, 50, None, None, None, None],
            tracked_index=[None, "VOO", None, None, "none", "none", None, "VOO", "none", "none", "none"],
        )
        none = np.array([0.00, 0.00, 0.00, 0.00, 0.20, 0.10, 0.00, 0.00, 0.05, 0.05, 0.00])

        def weights_for(**distance):
            row = np.full(len(labels), np.nan)
            for name, value in distance.items():
                row[labels.index(name)] = value
            return tactical_weights(targets, signals, row, cash_index=labels.index("SGOV"))[0]

        np.testing.assert_allclose(weights_for(AVES=0.30, CSVYX=0.10, VOO=0.20, DWAS=0.90), none + np.array([0, 0, 0, 0, 0, 0, 0.60, 0, 0, 0, 0]))
        np.testing.assert_allclose(weights_for(CSVYX=0.40, AVES=0.10, VOO=0.20), none + np.array([0, 0, 0.60, 0, 0, 0, 0, 0, 0, 0, 0]))
        np.testing.assert_allclose(
            weights_for(VOO=0.50, AVES=0.10, CSVYX=0.10),
            none + np.array([0.50, 0.00, 0, 0, 0, 0, 0, 0.10, 0, 0, 0]),
        )
        np.testing.assert_allclose(weights_for(AVES=-0.20, CSVYX=-0.05, VOO=-0.01), none + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.60]))
        self.assertAlmostEqual(weights_for(AVES=0.30).sum(), 1.0)


class SignalSimulationTest(unittest.TestCase):
    def test_a_never_liquidated_asset_still_triggers_and_the_closed_sleeve_earns_cash(self):
        signals = resolve_asset_signals(
            ["BTC", "TQQQ"],
            sma_exit=[200, 200],
            sma_enter=[200, 200],
            tracked_index=["none", None],
        )
        book = SignalBook(signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0)
        book.invested[:, 1] = False
        dt = 1 / 252
        drift = 0.10
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([0.80, 0.20]),
            drift=np.array([drift, drift]),
            volatility=np.array([0.0, 0.0]),
            T=2 * dt,
            dt=dt,
            num_paths=1,
            lower_threshold=np.array([0.50, 0.10]),
            upper_threshold=np.array([0.70, 0.50]),
            target_allocation=np.array([0.60, 0.40]),
            signal=book,
            rebalance_interval=1,
        )

        growth = np.exp(drift * dt)
        stepped = np.array([0.80 * growth, 0.20])
        np.testing.assert_allclose(prices[1, 0], stepped.sum() * np.array([0.60, 0.40]))
        self.assertGreater(upper_counts[0, 0], 0)
        np.testing.assert_array_equal(lower_counts, np.zeros((1, 2)))
        np.testing.assert_allclose(prices[2, 0, 0], prices[1, 0, 0] * growth)
        np.testing.assert_allclose(prices[2, 0, 1], prices[1, 0, 1])
        self.assertTrue(book.held()[0, 0])
        self.assertFalse(book.held()[0, 1])

    def test_monthly_momentum_assigns_a_different_sleeve_on_each_path(self):
        labels = ["VOO", "SPMO", "CSVYX", "DWAS", "DBMF", "BTC", "AVES", "TQQQ", "KMLM", "GLD", "SGOV"]
        targets = np.array([0.25, 0.00, 0.15, 0.00, 0.20, 0.10, 0.15, 0.05, 0.05, 0.05, 0.00])
        signals = resolve_asset_signals(
            labels,
            sma_exit=[2, None, 2, 2, None, None, 2, None, None, None, None],
            sma_enter=[2, None, 2, 2, None, None, 2, None, None, None, None],
            tracked_index=[None, "VOO", None, None, "none", "none", None, "VOO", "none", "none", "none"],
        )
        book = SignalBook(signals, num_paths=2, cash_drift=0.0, cash_volatility=0.0, cash_index=labels.index("SGOV"))
        growth = np.ones((2, len(labels)))
        growth[0, labels.index("AVES")] = 3.0
        growth[1, labels.index("VOO")] = 3.0
        book.observe(growth)

        prices = np.broadcast_to(100.0 * targets, (2, len(labels))).copy()
        drift = np.zeros_like(prices)
        drift[:, labels.index("AVES")] = 0.10
        dt = 1 / 252
        next_prices, momentum_targets, market_growth = monthly_momentum(
            prices, drift, np.zeros_like(prices), dt, book, targets,
        )

        none = np.array([0.00, 0.00, 0.00, 0.00, 0.20, 0.10, 0.00, 0.00, 0.05, 0.05, 0.00])
        expected = np.vstack([
            none + np.array([0, 0, 0, 0, 0, 0, 0.60, 0, 0, 0, 0]),
            none + np.array([0.50, 0.00, 0, 0, 0, 0, 0, 0.10, 0, 0, 0]),
        ])
        np.testing.assert_allclose(momentum_targets, expected)
        np.testing.assert_allclose(momentum_targets.sum(axis=1), [1.0, 1.0])
        aves = labels.index("AVES")
        asset_growth = np.exp(0.10 * dt)
        np.testing.assert_allclose(market_growth[:, aves], [asset_growth, asset_growth])
        np.testing.assert_allclose(next_prices[:, aves], [prices[0, aves] * asset_growth, prices[1, aves]])


class MomentumSleeveTest(unittest.TestCase):
    LABELS = ["VOO", "VXUS", "SGOV", "DBMF", "BTC"]
    TARGETS = np.array([0.55, 0.60, 0.00, 0.20, 0.20])

    def book(self, num_paths=1, cash_drift=0.05, exit_days=3, enter_days=2, trade_interval=25, beat_cash=True):
        signals = resolve_asset_signals(
            self.LABELS,
            sma_exit=[exit_days, exit_days, 200, None, None],
            sma_enter=[enter_days, enter_days, 200, None, None],
            asset_type=["Momentum", "Momentum", "Momentum", "Alternate", "Alternate"],
        )
        return SignalBook(
            signals,
            num_paths=num_paths,
            cash_drift=cash_drift,
            cash_volatility=0.0,
            cash_index=self.LABELS.index("SGOV"),
            trade_interval=trade_interval,
            beat_cash=beat_cash,
        )

    def load_prices(self, book, prices):
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            prices = prices[:, None, :]
        count = prices.shape[0]
        book._history = np.ones_like(book._history)
        book._history[:count] = prices
        book._count = count
        book._pos = count - 1
        book._sync_window_sums()

    def test_asset_type_marks_momentum_and_alternate_sleeves(self):
        book = self.book()

        self.assertEqual([signal.asset_type for signal in book.signals], [
            "momentum", "momentum", "momentum", "alternate", "alternate",
        ])

    def test_invested_book_takes_the_largest_exit_gap_only_when_it_beats_sgov(self):
        # Last three closes. A repeated close on the next day is not part of this check.
        # VOO is 10% above its exit average; VXUS is 20% above. The SGOV hurdle is 5%.
        voo = self._closes_for_gap(0.10)
        vxus = self._closes_for_gap(0.20)
        book = self.book()
        self.load_prices(book, np.column_stack([voo, vxus, np.ones(3), np.ones(3), np.ones(3)]))
        book.start(in_cash=False, asset=0)

        book.choose_momentum()

        self.assertFalse(book.in_cash[0])
        self.assertEqual(book.selection[0], 1)
        # Both names beat cash, so each takes one third of the 0.60 sleeve and the open third stays in SGOV.
        np.testing.assert_allclose(
            book.target_weights(self.TARGETS)[0],
            [0.20, 0.20, 0.20, 0.20, 0.20],
        )

        # 4% and 3% are both under the 5% SGOV rate, so the sleeve goes to cash.
        short = self.book()
        self.load_prices(short, np.column_stack([
            self._closes_for_gap(0.04),
            self._closes_for_gap(0.03),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        short.start(in_cash=False, asset=0)
        short.choose_momentum()

        self.assertTrue(short.in_cash[0])
        self.assertEqual(short.selection[0], 2)
        np.testing.assert_allclose(
            short.target_weights(self.TARGETS)[0],
            [0.0, 0.0, 0.60, 0.20, 0.20],
        )

    def test_beat_cash_off_holds_a_name_that_is_above_its_sma(self):
        # 4% and 3% sit under the 5% cash rate and above the average. The sleeve
        # takes both, and the larger gap is the lead name.
        book = self.book(beat_cash=False)
        self.load_prices(book, np.column_stack([
            self._closes_for_gap(0.04),
            self._closes_for_gap(0.03),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)

        book.choose_momentum()

        self.assertFalse(book.in_cash[0])
        self.assertEqual(book.selection[0], 0)
        np.testing.assert_allclose(
            book.target_weights(self.TARGETS)[0],
            [0.20, 0.20, 0.20, 0.20, 0.20],
        )

    def test_beat_cash_off_leaves_a_name_at_or_under_its_sma_in_cash(self):
        # A flat print ties the average. A lower print sits under it. Neither clears the SMA.
        book = self.book(beat_cash=False)
        self.load_prices(book, np.column_stack([
            self._closes_for_gap(0.0),
            self._closes_for_gap(-0.03),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)

        book.choose_momentum()

        self.assertTrue(book.in_cash[0])
        self.assertEqual(book.selection[0], 2)
        np.testing.assert_allclose(
            book.target_weights(self.TARGETS)[0],
            [0.0, 0.0, 0.60, 0.20, 0.20],
        )

    def test_cash_uses_the_enter_average_and_ignores_the_exit_average(self):
        # Exit gap is 1%, under the 5% hurdle. Enter gap on the same closes is 20%.
        prices = np.array([
            [1.955445544554455, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0, 1.0, 1.0],
        ])
        invested = self.book()
        self.load_prices(invested, prices)
        invested.start(in_cash=False, asset=0)
        invested.choose_momentum()
        self.assertTrue(invested.in_cash[0])
        self.assertEqual(invested.selection[0], 2)

        parked = self.book()
        self.load_prices(parked, prices)
        parked.start(in_cash=True)
        parked.choose_momentum()
        self.assertFalse(parked.in_cash[0])
        self.assertEqual(parked.selection[0], 0)

    def test_momentum_choice_waits_for_the_25th_day(self):
        book = self.book()
        self.load_prices(book, np.column_stack([
            self._closes_for_gap(0.04),
            self._closes_for_gap(0.20),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)

        for _ in range(24):
            self.assertFalse(book.consider_trade())
        self.assertEqual(book.selection[0], 0)

        self.assertTrue(book.consider_trade())
        self.assertEqual(book.selection[0], 1)

    def test_alternate_band_rebalances_to_the_current_momentum_and_alternate_targets(self):
        book = self.book(trade_interval=25)
        self.load_prices(book, np.column_stack([
            self._closes_for_gap(0.04),
            self._closes_for_gap(0.20),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([40.0, 0.0, 0.0, 20.0, 40.0]),
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.array([0.0, 0.0, 0.0, 0.05, 0.05]),
            upper_threshold=np.array([0.10, 0.10, 0.10, 0.90, 0.25]),
            target_allocation=self.TARGETS,
            signal=book,
            rebalance_interval=1,
        )

        np.testing.assert_allclose(prices[1, 0], [60.0, 0.0, 0.0, 20.0, 20.0])
        np.testing.assert_array_equal(upper_counts, [[0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(lower_counts, np.zeros((1, 5)))

    def test_momentum_day_moves_the_sleeve_without_resetting_alternates(self):
        book = self.book(trade_interval=1)
        self.load_prices(book, np.column_stack([
            self._closes_for_gap(0.10, close=1.2),
            self._closes_for_gap(0.20, close=1.5),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)
        drift = np.zeros(5)
        drift[3] = 0.50
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([50.0, 0.0, 0.0, 50.0, 0.0]),
            drift=drift,
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.full(5, -1.0),
            upper_threshold=np.full(5, 0.95),
            target_allocation=self.TARGETS,
            signal=book,
        )

        growth = np.exp(0.50 / 252)
        third = 50.0 / 3.0
        np.testing.assert_allclose(prices[1, 0], [third, third, third, 50.0 * growth, 0.0])
        np.testing.assert_array_equal(lower_counts + upper_counts, np.zeros((1, 5)))

    def test_a_momentum_band_does_not_trade_when_alternates_are_inside_theirs(self):
        book = self.book(trade_interval=25)
        book.start(in_cash=False, asset=0)
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([80.0, 0.0, 0.0, 20.0, 0.0]),
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.array([-1.0, -1.0, -1.0, 0.05, -1.0]),
            upper_threshold=np.array([0.30, 1.0, 1.0, 0.50, 1.0]),
            target_allocation=self.TARGETS,
            signal=book,
        )

        np.testing.assert_allclose(prices[1, 0], [80.0, 0.0, 0.0, 20.0, 0.0])
        np.testing.assert_array_equal(lower_counts + upper_counts, np.zeros((1, 5)))

    def test_voo_keeps_a_shadow_price_while_the_sleeve_sits_in_sgov(self):
        book = self.book(trade_interval=25)
        book.start(in_cash=True)
        drift = np.zeros(5)
        drift[0] = 0.10
        simulate_with_rebalance(
            S0=np.array([0.0, 0.0, 60.0, 20.0, 20.0]),
            drift=drift,
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.zeros(5),
            upper_threshold=np.ones(5),
            target_allocation=self.TARGETS,
            signal=book,
        )

        np.testing.assert_allclose(book._history[book._pos, 0, 0], np.exp(0.10 / 252))

    def test_momentum_rotation_still_runs_between_band_checks(self):
        book = self.book(trade_interval=1)
        self.load_prices(book, np.column_stack([
            self._closes_for_gap(0.10, close=1.2),
            self._closes_for_gap(0.20, close=1.5),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)
        drift = np.zeros(5)
        drift[3] = 0.50
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([50.0, 0.0, 0.0, 50.0, 0.0]),
            drift=drift,
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.full(5, -1.0),
            upper_threshold=np.full(5, 0.95),
            target_allocation=self.TARGETS,
            signal=book,
            rebalance_interval=2,
        )

        growth = np.exp(0.50 / 252)
        third = 50.0 / 3.0
        np.testing.assert_allclose(prices[1, 0], [third, third, third, 50.0 * growth, 0.0])
        np.testing.assert_array_equal(lower_counts + upper_counts, np.zeros((1, 5)))

    def test_momentum_shift_rotates_the_sleeve_without_consulting_bands(self):
        book = self.book(trade_interval=1)
        self.load_prices(book, np.column_stack([
            self._closes_for_gap(0.10, close=1.2),
            self._closes_for_gap(0.20, close=1.5),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)
        growth = np.array([[1.0, 1.0, 1.0, 1.1, 1.0]])

        shifted, traded = momentum_shift(np.array([[50.0, 0.0, 0.0, 50.0, 0.0]]), book, growth)

        self.assertTrue(traded)
        self.assertEqual(book.selection[0], 1)
        third = 50.0 / 3.0
        np.testing.assert_allclose(shifted[0], [third, third, third, 55.0, 0.0])

    def test_momentum_shift_keeps_the_book_until_the_trade_day(self):
        book = self.book(trade_interval=25)
        self.load_prices(book, np.column_stack([
            self._closes_for_gap(0.04),
            self._closes_for_gap(0.20),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)
        holdings = np.array([[50.0, 0.0, 0.0, 30.0, 20.0]])

        shifted, traded = momentum_shift(holdings, book, np.ones((1, 5)))

        self.assertFalse(traded)
        self.assertEqual(book.selection[0], 0)
        np.testing.assert_allclose(shifted, holdings)

    def test_dynamic_rebalance_snaps_targets_without_a_momentum_signal(self):
        # DBMF is outside its band. The targets still name VOO, so the snap does not rotate to VXUS.
        valuations = np.array([[50.0, 0.0, 0.0, 55.0, 0.0]])
        book, lower_breach, upper_breach = dynamic_rebalance(
            valuations,
            lower_threshold=np.array([-1.0, -1.0, -1.0, 0.05, -1.0]),
            upper_threshold=np.array([1.0, 1.0, 1.0, 0.40, 1.0]),
            target_weights=np.array([0.60, 0.0, 0.0, 0.20, 0.20]),
            trigger_mask=np.array([False, False, False, True, True]),
        )

        np.testing.assert_allclose(book[0], [63.0, 0.0, 0.0, 21.0, 21.0])
        np.testing.assert_array_equal(upper_breach, [[False, False, False, True, False]])
        np.testing.assert_array_equal(lower_breach, np.zeros((1, 5)))

    def test_the_average_drops_the_most_recent_excluded_closes(self):
        # Five closes, a three-day average, and a two-day exclusion.
        # The average is the three closes before the last two: 10, 10, and 10.
        # A straight three-day average would be (10 + 2 + 6) / 3 = 6, and 6 is not below 6.
        signals = resolve_asset_signals(
            ["VOO"],
            sma_exit=[3],
            sma_enter=[3],
            sma_exclusion=[2],
        )
        book = SignalBook(signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0)
        prices = np.array([10.0, 10.0, 10.0, 2.0, 6.0])
        book._history = np.ones_like(book._history)
        book._history[: prices.shape[0], 0, 0] = prices
        book._count = prices.shape[0]
        book._pos = prices.shape[0] - 1
        book._sync_window_sums()

        np.testing.assert_allclose(book._window_mean(book._exit_sum, book._exit_lengths), [[10.0]])
        book._update()
        self.assertFalse(book.invested[0, 0])

    def test_the_average_waits_until_the_excluded_window_is_full(self):
        signals = resolve_asset_signals(
            ["VOO"],
            sma_exit=[3],
            sma_enter=[3],
            sma_exclusion=[2],
        )
        book = SignalBook(signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0)
        prices = np.array([10.0, 10.0, 10.0, 1.0])
        book._history = np.ones_like(book._history)
        book._history[: prices.shape[0], 0, 0] = prices
        book._count = prices.shape[0]
        book._pos = prices.shape[0] - 1
        book._steps_seen = prices.shape[0]
        book._sync_window_sums()

        book._update()

        self.assertTrue(book.invested[0, 0])

    def test_running_excluded_mean_matches_the_older_closes(self):
        signals = resolve_asset_signals(
            ["VOO", "SPMO"],
            sma_exit=[5, 3],
            sma_enter=[4, 2],
            sma_exclusion=[2, 1],
        )
        book = SignalBook(signals, num_paths=2, cash_drift=0.0, cash_volatility=0.0)
        rng = np.random.default_rng(2)
        for _ in range(16):
            book.observe(rng.uniform(0.97, 1.04, size=(2, 2)))
            for lengths, spans, sums in (
                (book._exit_lengths, book._exit_spans, book._exit_sum),
                (book._enter_lengths, book._enter_spans, book._enter_sum),
            ):
                for asset, length in enumerate(lengths):
                    exclusion = int(book._exclusion[asset])
                    if book._count < int(spans[asset]):
                        continue
                    window = book._window(int(length) + exclusion)[:, :, asset]
                    older = window if exclusion == 0 else window[:-exclusion]
                    np.testing.assert_allclose(
                        book._window_mean(sums, lengths)[:, asset],
                        older.mean(axis=0),
                        rtol=1e-8,
                        atol=1e-8,
                    )

    def test_running_window_mean_matches_the_gathered_window(self):
        signals = resolve_asset_signals(
            ["VOO", "SPMO", "SGOV"],
            sma_exit=[5, 3, 4],
            sma_enter=[4, 2, 1],
            tracked_index=[None, "VOO", "none"],
        )
        book = SignalBook(signals, num_paths=3, cash_drift=0.0, cash_volatility=0.0, cash_index=2)
        rng = np.random.default_rng(1)
        for _ in range(12):
            book.observe(rng.uniform(0.97, 1.04, size=(3, 3)))
            for lengths, sums in ((book._exit_lengths, book._exit_sum), (book._enter_lengths, book._enter_sum)):
                for asset, length in enumerate(lengths):
                    if book._count < length:
                        continue
                    expected = book._window(int(length))[:, :, asset].mean(axis=0)
                    np.testing.assert_allclose(sums[:, asset] / length, expected, rtol=1e-8, atol=1e-8)

    @staticmethod
    def _closes_for_gap(gap, close=None):
        # Three closes whose last print sits `gap` above the three-day average.
        if close is None:
            close = (1.0 + gap) * 2.0 / (2.0 - gap)
        return np.array([1.0, 1.0, close])


class BucketMomentumTest(unittest.TestCase):
    """A named bucket is one momentum candidate. Its gap is the weighted average of its members."""

    LABELS = ["XLK", "XLY", "XLE", "SGOV", "DBMF"]
    TARGETS = np.array([0.50, 0.30, 0.80, 0.80, 0.20])

    def book(self, buckets=("Cyclical", "Cyclical", None, None, None), num_paths=1, cash_drift=0.05, top_num=3):
        signals = resolve_asset_signals(
            self.LABELS,
            sma_exit=[3, 3, 3, 200, None],
            sma_enter=[2, 2, 2, 200, None],
            asset_type=["Momentum", "Momentum", "Momentum", "Momentum", "Alternate"],
            bucket=buckets,
        )
        return SignalBook(
            signals,
            num_paths=num_paths,
            cash_drift=cash_drift,
            cash_volatility=0.0,
            cash_index=self.LABELS.index("SGOV"),
            trade_interval=1,
            top_num=top_num,
        )

    def load_prices(self, book, prices):
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            prices = prices[:, None, :]
        count = prices.shape[0]
        book._history = np.ones_like(book._history)
        book._history[:count] = prices
        book._count = count
        book._pos = count - 1
        book._sync_window_sums()

    def prices_for(self, gaps):
        columns = [
            MomentumSleeveTest._closes_for_gap(gaps[0]),
            MomentumSleeveTest._closes_for_gap(gaps[1]),
            MomentumSleeveTest._closes_for_gap(gaps[2]),
            np.ones(3),
            np.ones(3),
        ]
        return np.column_stack(columns)

    def test_blank_bucket_cells_leave_each_momentum_name_on_its_own(self):
        signals = resolve_asset_signals(
            ["XLK", "XLY", "SGOV"],
            sma_exit=[3, 3, 3],
            sma_enter=[3, 3, 3],
            asset_type=["Momentum", "Momentum", "Momentum"],
            bucket=[None, "", "  "],
        )

        self.assertEqual([signal.bucket for signal in signals], [None, None, None])

    def test_bucket_score_is_the_allocation_weighted_gap_and_the_sleeve_splits_across_members(self):
        # Cyclical: 0.50 * 40% + 0.30 * 20% = 32.5% over the average. XLE is only 10% over.
        book = self.book()
        self.load_prices(book, self.prices_for([0.40, 0.20, 0.10]))
        book.start(in_cash=False, asset=2)

        book.choose_momentum(self.TARGETS)

        self.assertFalse(book.in_cash[0])
        self.assertEqual(book.selected_name(self.LABELS), "Cyclical")
        third = 0.80 / 3.0
        np.testing.assert_allclose(
            book.target_weights(self.TARGETS)[0],
            [0.50 / 0.80 * third, 0.30 / 0.80 * third, third, third, 0.20],
        )

    def test_a_solo_name_beats_a_bucket_whose_weighted_gap_is_smaller(self):
        # Cyclical's weighted gap is 2%, under the 5% hurdle. XLE is 20% over.
        book = self.book()
        self.load_prices(book, self.prices_for([0.02, 0.02, 0.20]))
        book.start(in_cash=False, asset=0)

        book.choose_momentum(self.TARGETS)

        third = 0.80 / 3.0
        self.assertEqual(book.selected_name(self.LABELS), "XLE")
        np.testing.assert_allclose(
            book.target_weights(self.TARGETS)[0],
            [0.0, 0.0, third, 2.0 * third, 0.20],
        )

    def test_two_paths_can_pick_a_bucket_and_a_solo_name_on_the_same_trade(self):
        book = self.book(num_paths=2)
        path_bucket = self.prices_for([0.40, 0.20, 0.10])
        path_solo = self.prices_for([0.02, 0.02, 0.30])
        self.load_prices(book, np.stack([path_bucket, path_solo], axis=1))
        book.start(in_cash=False, asset=2)

        book.choose_momentum(self.TARGETS)

        third = 0.80 / 3.0
        self.assertEqual(book.selected_names(self.LABELS), ["Cyclical", "XLE"])
        self.assertEqual(int(book.selection[1]), 2)
        np.testing.assert_allclose(
            book.target_weights(self.TARGETS),
            [
                [0.50 / 0.80 * third, 0.30 / 0.80 * third, third, third, 0.20],
                [0.0, 0.0, third, 2.0 * third, 0.20],
            ],
        )

    def test_a_bucket_under_the_cash_rate_parks_the_sleeve(self):
        book = self.book()
        self.load_prices(book, self.prices_for([0.04, 0.04, 0.03]))
        book.start(in_cash=False, asset=2)

        book.choose_momentum(self.TARGETS)

        self.assertTrue(book.in_cash[0])
        self.assertEqual(book.selected_name(self.LABELS), "SGOV")
        np.testing.assert_allclose(
            book.target_weights(self.TARGETS)[0],
            [0.0, 0.0, 0.0, 0.80, 0.20],
        )

    def test_bucket_allocations_must_add_up_to_the_momentum_portion(self):
        book = self.book()
        short = np.array([0.40, 0.20, 0.80, 0.80, 0.20])

        with self.assertRaises(ValueError) as caught:
            book.target_weights(short)

        message = str(caught.exception)
        self.assertIn("Cyclical", message)
        self.assertIn("momentum portion", message)

    def test_a_trade_splits_the_sleeve_across_the_bucket(self):
        book = self.book()
        self.load_prices(book, self.prices_for([0.50, 0.50, 0.30]))
        book.start(in_cash=False, asset=2)

        shifted, traded = momentum_shift(
            np.array([[0.0, 0.0, 80.0, 0.0, 20.0]]),
            book,
            np.ones((1, 5)),
            self.TARGETS,
        )

        self.assertTrue(traded)
        self.assertEqual(book.selected_names(self.LABELS), ["Cyclical", "XLE"])
        third = 80.0 / 3.0
        np.testing.assert_allclose(shifted[0], [0.50 / 0.80 * third, 0.30 / 0.80 * third, third, third, 20.0])

    def test_only_the_top_three_names_take_a_third_of_the_sleeve(self):
        labels = ["AAA", "BBB", "CCC", "DDD", "SGOV"]
        signals = resolve_asset_signals(
            labels,
            sma_exit=[3, 3, 3, 3, 3],
            sma_enter=[3, 3, 3, 3, 3],
            asset_type=["Momentum", "Momentum", "Momentum", "Momentum", "Momentum"],
        )
        book = SignalBook(signals, num_paths=1, cash_drift=0.05, cash_volatility=0.0, cash_index=4, trade_interval=1)
        gaps = [0.40, 0.30, 0.20, 0.10]
        prices = np.column_stack([MomentumSleeveTest._closes_for_gap(gap) for gap in gaps] + [np.ones(3)])
        self.load_prices(book, prices)
        book.start(in_cash=True)
        targets = np.ones(5)

        book.choose_momentum(targets)

        self.assertEqual(book.selected_names(labels), ["AAA", "BBB", "CCC"])
        self.assertFalse(book.in_cash[0])
        np.testing.assert_allclose(book.target_weights(targets)[0], [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0])

    def test_top_num_keeps_only_that_many_solo_names(self):
        labels = ["AAA", "BBB", "CCC", "DDD", "SGOV"]
        signals = resolve_asset_signals(
            labels,
            sma_exit=[3, 3, 3, 3, 3],
            sma_enter=[3, 3, 3, 3, 3],
            asset_type=["Momentum", "Momentum", "Momentum", "Momentum", "Momentum"],
        )
        book = SignalBook(
            signals, num_paths=1, cash_drift=0.05, cash_volatility=0.0, cash_index=4, trade_interval=1, top_num=2
        )
        gaps = [0.40, 0.30, 0.20, 0.10]
        prices = np.column_stack([MomentumSleeveTest._closes_for_gap(gap) for gap in gaps] + [np.ones(3)])
        self.load_prices(book, prices)
        book.start(in_cash=True)

        book.choose_momentum(np.ones(5))

        self.assertEqual(book.selected_names(labels), ["AAA", "BBB"])
        np.testing.assert_allclose(book.target_weights(np.ones(5))[0], [0.5, 0.5, 0.0, 0.0, 0.0])

    def test_top_num_counts_a_bucket_as_one_pick(self):
        book = self.book(top_num=1)
        self.load_prices(book, self.prices_for([0.40, 0.20, 0.10]))
        book.start(in_cash=False, asset=2)

        book.choose_momentum(self.TARGETS)

        self.assertEqual(book.selected_names(self.LABELS), ["Cyclical"])
        np.testing.assert_allclose(book.target_weights(self.TARGETS)[0], [0.50, 0.30, 0.0, 0.0, 0.20])

    def test_two_picks_split_the_sleeve_between_a_bucket_and_a_solo_name(self):
        book = self.book(top_num=2)
        self.load_prices(book, self.prices_for([0.40, 0.20, 0.10]))
        book.start(in_cash=False, asset=2)

        book.choose_momentum(self.TARGETS)

        half = 0.80 / 2.0
        self.assertEqual(book.selected_names(self.LABELS), ["Cyclical", "XLE"])
        np.testing.assert_allclose(
            book.target_weights(self.TARGETS)[0],
            [0.50 / 0.80 * half, 0.30 / 0.80 * half, half, 0.0, 0.20],
        )

    def test_top_num_must_be_a_positive_whole_number(self):
        signals = resolve_asset_signals(
            self.LABELS,
            sma_exit=[3, 3, 3, 200, None],
            sma_enter=[2, 2, 2, 200, None],
            asset_type=["Momentum", "Momentum", "Momentum", "Momentum", "Alternate"],
        )
        with self.assertRaises(ValueError):
            SignalBook(signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0, cash_index=3, top_num=0)

    def test_sgov_and_non_momentum_names_cannot_join_a_bucket(self):
        cash_signals = resolve_asset_signals(
            self.LABELS,
            sma_exit=[3, 3, 3, 3, None],
            sma_enter=[3, 3, 3, 3, None],
            asset_type=["Momentum", "Momentum", "Momentum", "Momentum", "Alternate"],
            bucket=["Cyclical", None, None, "Cyclical", None],
        )
        with self.assertRaises(ValueError):
            SignalBook(cash_signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0, cash_index=3)

        alternate_signals = resolve_asset_signals(
            self.LABELS,
            sma_exit=[3, 3, 3, 3, None],
            sma_enter=[3, 3, 3, 3, None],
            asset_type=["Momentum", "Momentum", "Momentum", "Momentum", "Alternate"],
            bucket=[None, None, None, None, "Hedge"],
        )
        with self.assertRaises(ValueError):
            SignalBook(alternate_signals, num_paths=1, cash_drift=0.0, cash_volatility=0.0, cash_index=3)


class GatedSleeveTest(unittest.TestCase):
    """A gated asset is an alternate that sits in SGOV while the momentum sleeve does."""

    LABELS = ["VOO", "VXUS", "SGOV", "DBMF", "BTC"]
    TARGETS = np.array([0.55, 0.60, 0.00, 0.20, 0.20])

    def book(self, num_paths=1, cash_drift=0.05, exit_days=3, enter_days=2, trade_interval=25):
        signals = resolve_asset_signals(
            self.LABELS,
            sma_exit=[exit_days, exit_days, 200, None, None],
            sma_enter=[enter_days, enter_days, 200, None, None],
            asset_type=["Momentum", "Momentum", "Momentum", "Alternate", "Gated"],
        )
        return SignalBook(
            signals,
            num_paths=num_paths,
            cash_drift=cash_drift,
            cash_volatility=0.0,
            cash_index=self.LABELS.index("SGOV"),
            trade_interval=trade_interval,
        )

    def load_prices(self, book, prices):
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            prices = prices[:, None, :]
        count = prices.shape[0]
        book._history = np.ones_like(book._history)
        book._history[:count] = prices
        book._count = count
        book._pos = count - 1
        book._sync_window_sums()

    def test_gated_counts_as_an_alternate_that_parks_with_cash(self):
        book = self.book()

        self.assertEqual([signal.asset_type for signal in book.signals], [
            "momentum", "momentum", "momentum", "alternate", "gated",
        ])
        np.testing.assert_array_equal(book.gated_mask, [False, False, False, False, True])
        np.testing.assert_array_equal(book.alternate_mask, [False, False, False, True, True])

    def test_cash_targets_move_the_gated_weight_onto_sgov(self):
        book = self.book()
        book.start(in_cash=True)

        np.testing.assert_allclose(book.target_weights(self.TARGETS)[0], [0.0, 0.0, 0.80, 0.20, 0.0])

        book.start(in_cash=False, asset=1)
        np.testing.assert_allclose(book.target_weights(self.TARGETS)[0], [0.0, 0.60, 0.0, 0.20, 0.20])

    def test_two_paths_park_and_invest_the_gated_sleeve_separately(self):
        book = self.book(num_paths=2)
        book.start(in_cash=True)
        book.in_cash[:] = [True, False]
        book.selection[:] = [2, 1]

        np.testing.assert_allclose(
            book.target_weights(self.TARGETS),
            [
                [0.0, 0.0, 0.80, 0.20, 0.0],
                [0.0, 0.60, 0.0, 0.20, 0.20],
            ],
        )

    def test_the_trade_into_sgov_moves_gated_dollars_and_leaves_dbmf(self):
        # 4% and 3% stay under the 5% hurdle after the flat mark, so the sleeve goes to cash.
        book = self.book(trade_interval=1)
        self.load_prices(book, np.column_stack([
            MomentumSleeveTest._closes_for_gap(0.04),
            MomentumSleeveTest._closes_for_gap(0.03),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)

        shifted, traded = momentum_shift(
            np.array([[50.0, 0.0, 0.0, 30.0, 20.0]]),
            book,
            np.ones((1, 5)),
            self.TARGETS,
        )

        self.assertTrue(traded)
        self.assertTrue(book.in_cash[0])
        np.testing.assert_allclose(shifted[0], [0.0, 0.0, 70.0, 30.0, 0.0])

    def test_nothing_that_beats_sgov_moves_the_sleeve_and_btc_into_cash(self):
        # Both equity gaps sit under the cash rate, so SGOV wins the whole sleeve.
        # BTC is gated, and an activated SGOV position takes that cash too. DBMF stays.
        book = self.book(trade_interval=1)
        self.load_prices(book, np.column_stack([
            MomentumSleeveTest._closes_for_gap(0.04),
            MomentumSleeveTest._closes_for_gap(0.03),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)

        prices, _lower, _upper = simulate_with_rebalance(
            S0=np.array([50.0, 0.0, 0.0, 30.0, 20.0]),
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.zeros(5),
            upper_threshold=np.ones(5),
            target_allocation=self.TARGETS,
            signal=book,
            rebalance_interval=21,
        )

        self.assertTrue(book.in_cash[0])
        self.assertEqual(book.selected_names(self.LABELS), ["SGOV"])
        np.testing.assert_allclose(prices[1, 0], [0.0, 0.0, 70.0, 30.0, 0.0])
        np.testing.assert_allclose(book.target_weights(self.TARGETS)[0], [0.0, 0.0, 0.80, 0.20, 0.0])

    def test_a_band_snap_on_the_cash_trade_still_parks_btc_in_sgov(self):
        # DBMF is outside its band, so the day snaps to targets. Those targets
        # already hold BTC inside the activated SGOV pile.
        book = self.book(trade_interval=1)
        self.load_prices(book, np.column_stack([
            MomentumSleeveTest._closes_for_gap(0.04),
            MomentumSleeveTest._closes_for_gap(0.03),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)

        prices, _lower, upper = simulate_with_rebalance(
            S0=np.array([50.0, 0.0, 0.0, 50.0, 20.0]),
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.array([0.0, 0.0, 0.0, 0.05, 0.02]),
            upper_threshold=np.array([1.0, 1.0, 1.0, 0.35, 0.20]),
            target_allocation=self.TARGETS,
            signal=book,
            rebalance_interval=1,
        )

        self.assertTrue(book.in_cash[0])
        np.testing.assert_array_equal(upper[0] > 0, [False, False, False, True, False])
        np.testing.assert_allclose(prices[1, 0], [0.0, 0.0, 96.0, 24.0, 0.0])

    def test_one_qualifying_name_keeps_its_third_and_leaves_the_gated_asset_invested(self):
        # VOO clears the hurdle and VXUS does not, so two thirds of the momentum weight sit in SGOV.
        # Cash did not win the book, so BTC stays at its target instead of joining SGOV.
        book = self.book()
        self.load_prices(book, np.column_stack([
            MomentumSleeveTest._closes_for_gap(0.20),
            MomentumSleeveTest._closes_for_gap(0.01),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ]))
        book.start(in_cash=False, asset=0)

        book.choose_momentum(self.TARGETS)

        self.assertFalse(book.in_cash[0])
        np.testing.assert_allclose(
            book.target_weights(self.TARGETS)[0],
            [0.20, 0.0, 0.40, 0.20, 0.20],
        )

    def test_the_trade_out_of_sgov_splits_the_cash_pile(self):
        # Two flat closes, then VOO's market print is 1.2. Its two-day enter gap is about 9%.
        book = self.book(trade_interval=1)
        self.load_prices(book, np.ones((2, 5)))
        book.start(in_cash=True)
        growth = np.array([[1.2, 1.0, 1.0, 1.0, 1.0]])

        shifted, traded = momentum_shift(
            np.array([[0.0, 0.0, 80.0, 20.0, 0.0]]),
            book,
            growth,
            self.TARGETS,
        )

        self.assertTrue(traded)
        self.assertFalse(book.in_cash[0])
        self.assertEqual(book.selection[0], 0)
        np.testing.assert_allclose(shifted[0], [20.0, 0.0, 40.0, 20.0, 20.0])

        # DBMF has drifted to 40. It stays there. BTC takes 20/80 of the cash pile that is left.
        drifted = self.book(trade_interval=1)
        self.load_prices(drifted, np.ones((2, 5)))
        drifted.start(in_cash=True)
        shifted, traded = momentum_shift(
            np.array([[0.0, 0.0, 60.0, 40.0, 0.0]]),
            drifted,
            growth,
            self.TARGETS,
        )

        self.assertTrue(traded)
        np.testing.assert_allclose(shifted[0], [15.0, 0.0, 30.0, 40.0, 15.0])

    def test_two_paths_can_park_and_keep_the_gated_holding_on_the_same_trade(self):
        book = self.book(num_paths=2, trade_interval=1)
        cash_path = np.column_stack([
            MomentumSleeveTest._closes_for_gap(0.04),
            MomentumSleeveTest._closes_for_gap(0.03),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ])
        invested_path = np.column_stack([
            MomentumSleeveTest._closes_for_gap(0.10, close=1.2),
            MomentumSleeveTest._closes_for_gap(0.20, close=1.5),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ])
        self.load_prices(book, np.stack([cash_path, invested_path], axis=1))
        book.start(in_cash=False, asset=0)
        holdings = np.array([
            [50.0, 0.0, 0.0, 30.0, 20.0],
            [50.0, 0.0, 0.0, 30.0, 20.0],
        ])

        shifted, traded = momentum_shift(holdings, book, np.ones((2, 5)), self.TARGETS)

        self.assertTrue(traded)
        self.assertTrue(book.in_cash[0])
        self.assertFalse(book.in_cash[1])
        np.testing.assert_allclose(shifted[0], [0.0, 0.0, 70.0, 30.0, 0.0])
        third = 50.0 / 3.0
        np.testing.assert_allclose(shifted[1], [third, third, third, 30.0, 20.0])

    def test_a_parked_gated_weight_does_not_rebalance_dbmf(self):
        book = self.book(trade_interval=25)
        book.start(in_cash=True)
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([0.0, 0.0, 70.0, 30.0, 0.0]),
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.array([0.0, 0.0, 0.0, 0.05, 0.04]),
            upper_threshold=np.array([1.0, 1.0, 1.0, 0.50, 0.15]),
            target_allocation=self.TARGETS,
            signal=book,
        )

        np.testing.assert_allclose(prices[1, 0], [0.0, 0.0, 70.0, 30.0, 0.0])
        np.testing.assert_array_equal(lower_counts + upper_counts, np.zeros((1, 5)))

    def test_a_dbmf_band_while_parked_snaps_targets_and_keeps_btc_in_cash(self):
        book = self.book(trade_interval=25)
        book.start(in_cash=True)
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([0.0, 0.0, 50.0, 50.0, 0.0]),
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.array([0.0, 0.0, 0.0, 0.05, 0.04]),
            upper_threshold=np.array([1.0, 1.0, 1.0, 0.35, 0.15]),
            target_allocation=self.TARGETS,
            signal=book,
            rebalance_interval=1,
        )

        np.testing.assert_allclose(prices[1, 0], [0.0, 0.0, 80.0, 20.0, 0.0])
        np.testing.assert_array_equal(upper_counts, [[0, 0, 0, 1, 0]])
        np.testing.assert_array_equal(lower_counts, np.zeros((1, 5)))

    def test_an_invested_gated_asset_still_rebalances_on_its_band(self):
        book = self.book(trade_interval=25)
        book.start(in_cash=False, asset=0)
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([40.0, 0.0, 0.0, 20.0, 40.0]),
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.array([0.0, 0.0, 0.0, 0.05, 0.05]),
            upper_threshold=np.array([0.10, 0.10, 0.10, 0.90, 0.25]),
            target_allocation=self.TARGETS,
            signal=book,
            rebalance_interval=1,
        )

        np.testing.assert_allclose(prices[1, 0], [60.0, 0.0, 0.0, 20.0, 20.0])
        np.testing.assert_array_equal(upper_counts, [[0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(lower_counts, np.zeros((1, 5)))

    def test_the_opening_book_parks_a_gated_asset_when_the_sleeve_starts_in_sgov(self):
        book = self.book(trade_interval=25)
        prices, lower_counts, upper_counts = simulate_with_rebalance(
            S0=np.array([60.0, 0.0, 0.0, 20.0, 20.0]),
            drift=np.zeros(5),
            volatility=np.zeros(5),
            T=1 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.zeros(5),
            upper_threshold=np.ones(5),
            target_allocation=self.TARGETS,
            signal=book,
        )

        np.testing.assert_allclose(prices[0, 0], [0.0, 0.0, 80.0, 20.0, 0.0])
        np.testing.assert_array_equal(lower_counts + upper_counts, np.zeros((1, 5)))

    def test_a_per_path_trigger_mask_ignores_a_zero_target(self):
        valuations = np.array([
            [0.0, 0.0, 70.0, 30.0, 0.0],
            [40.0, 0.0, 0.0, 20.0, 40.0],
        ])
        targets = np.array([
            [0.0, 0.0, 0.80, 0.20, 0.0],
            [0.60, 0.0, 0.0, 0.20, 0.20],
        ])
        mask = np.array([
            [False, False, False, True, False],
            [False, False, False, True, True],
        ])

        rebalanced, lower_breach, upper_breach = dynamic_rebalance(
            valuations,
            lower_threshold=np.array([0.0, 0.0, 0.0, 0.05, 0.04]),
            upper_threshold=np.array([1.0, 1.0, 1.0, 0.50, 0.25]),
            target_weights=targets,
            trigger_mask=mask,
        )

        np.testing.assert_allclose(rebalanced[0], valuations[0])
        np.testing.assert_allclose(rebalanced[1], [60.0, 0.0, 0.0, 20.0, 20.0])
        np.testing.assert_array_equal(lower_breach, np.zeros((2, 5), dtype=bool))
        np.testing.assert_array_equal(upper_breach[0], [False, False, False, False, False])
        np.testing.assert_array_equal(upper_breach[1], [False, False, False, False, True])


class CurrentMomentumBucketTest(unittest.TestCase):
    LABELS = ["VOO", "VXUS", "SGOV"]

    def recommendation(self, prices, enter=3, exclusion=None, cash_drift=0.05, labels=None, asset_type=None, download=None):
        labels = self.LABELS if labels is None else labels
        kinds = ["Momentum", "Momentum", "Momentum"] if asset_type is None else asset_type
        return current_momentum_bucket(
            labels,
            sma_exit=[enter] * len(labels),
            sma_enter=[enter] * len(labels),
            cash_drift=cash_drift,
            asset_type=kinds,
            sma_exclusion=None if exclusion is None else [exclusion] * len(labels),
            prices=prices,
            download=download,
        )

    def test_cash_sleeve_recommends_the_widest_enter_gap_above_the_sgov_rate(self):
        prices = pd.DataFrame({
            "VOO": MomentumSleeveTest._closes_for_gap(0.20),
            "VXUS": MomentumSleeveTest._closes_for_gap(0.10),
            "SGOV": np.ones(3),
        })

        self.assertEqual(self.recommendation(prices), "VOO")

    def test_sleeve_stays_in_sgov_when_every_enter_gap_is_under_the_cash_rate(self):
        prices = pd.DataFrame({
            "VOO": MomentumSleeveTest._closes_for_gap(0.04),
            "VXUS": MomentumSleeveTest._closes_for_gap(0.03),
            "SGOV": np.ones(3),
        })

        self.assertEqual(self.recommendation(prices), "SGOV")

    def test_only_the_newest_enter_window_is_scored(self):
        # The oldest three closes would pick VXUS. The newest three pick VOO.
        prices = pd.DataFrame({
            "VOO": np.concatenate([
                MomentumSleeveTest._closes_for_gap(0.10),
                MomentumSleeveTest._closes_for_gap(0.20),
            ]),
            "VXUS": np.concatenate([
                MomentumSleeveTest._closes_for_gap(0.20),
                MomentumSleeveTest._closes_for_gap(0.10),
            ]),
            "SGOV": np.ones(6),
        })

        self.assertEqual(self.recommendation(prices), "VOO")

    def test_enter_history_keeps_the_closes_the_average_leaves_out(self):
        # Enter is three closes and the two newest stay out of the average.
        # VOO's last print is 20% above that older average. Three closes cannot fill it.
        prices = pd.DataFrame({
            "VOO": [10.0, 10.0, 10.0, 10.0, 12.0],
            "VXUS": [10.0, 10.0, 10.0, 10.0, 11.0],
            "SGOV": [1.0, 1.0, 1.0, 1.0, 1.0],
        })

        self.assertEqual(self.recommendation(prices, exclusion=2), "VOO")

    def test_a_short_history_stays_in_sgov(self):
        prices = pd.DataFrame({
            "VOO": MomentumSleeveTest._closes_for_gap(0.20),
            "VXUS": MomentumSleeveTest._closes_for_gap(0.10),
            "SGOV": np.ones(3),
        })

        self.assertEqual(self.recommendation(prices, enter=5), "SGOV")

    def test_an_alternate_gap_does_not_take_the_sleeve(self):
        prices = pd.DataFrame({
            "VOO": MomentumSleeveTest._closes_for_gap(0.04),
            "DBMF": MomentumSleeveTest._closes_for_gap(0.50),
            "SGOV": np.ones(3),
        })

        recommendation = self.recommendation(
            prices,
            labels=["VOO", "DBMF", "SGOV"],
            asset_type=["Momentum", "Alternate", "Momentum"],
        )

        self.assertEqual(recommendation, "SGOV")

    def test_download_uses_the_market_symbol_for_each_sleeve(self):
        requested = {}

        def download(tickers):
            requested["tickers"] = list(tickers)
            return pd.DataFrame({
                "VOO": MomentumSleeveTest._closes_for_gap(0.20),
                "BTC": MomentumSleeveTest._closes_for_gap(0.04),
                "SGOV": np.ones(3),
            })

        recommendation = self.recommendation(
            prices=None,
            labels=["VOO/VIIX", "BTC", "SGOV"],
            download=download,
        )

        self.assertEqual(requested["tickers"], ["VOO", "BTC", "SGOV"])
        self.assertEqual(recommendation, "VOO/VIIX")


if __name__ == "__main__":
    unittest.main()
