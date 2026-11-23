import tempfile
import unittest
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from gbm.regime import (
    DEFAULT_TRANSITION,
    DEFAULT_TRANSITION_INTERVAL,
    DEFAULT_TRANSITION_NOTES,
    SMA_PROBABILITY_PER_STD,
    SMA_Z_CAP,
    TRANSITION_INTERVAL_LABEL,
    TRANSITION_PER_STD_LABEL,
    TRANSITION_SHEET,
    TRANSITION_Z_CAP_LABEL,
    RegimeBook,
    add_transition_sheet,
    align_regime_parameters,
    churn_volatility_addon,
    draw_next_states,
    load_transition_matrix,
    sma_zscore,
    tilt_transition,
    validate_transition_matrix,
)
from gbm.gbm_step import simulate_with_rebalance

WORKBOOK = Path(__file__).resolve().parents[1] / "portfolio_gbm_calculator.xlsx"


class _Draws:
    """Uniforms returned in order, so a transition test can name the next regime."""

    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def random(self, size=None):
        count = 1 if size is None else int(np.prod(size))
        drawn = np.array(self.values[self.index:self.index + count], dtype=float)
        if drawn.size != count:
            raise AssertionError("the test ran out of uniform draws")
        self.index += count
        if size is None:
            return float(drawn[0])
        return drawn.reshape(size)


class RegimeTransitionTest(unittest.TestCase):
    def test_each_row_picks_the_next_regime_and_a_zero_probability_is_skipped(self):
        matrix = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.2, 0.1, 0.7],
        ])
        states = np.array([1, 2, 2])

        chosen = draw_next_states(states, matrix, _Draws([0.0, 0.49, 0.50]))

        self.assertEqual(chosen.tolist(), [3, 1, 2])

    def test_a_row_that_does_not_sum_to_one_is_rejected(self):
        matrix = np.array([
            [0.5, 0.5, 0.1],
            [0.15, 0.70, 0.15],
            [0.20, 0.10, 0.70],
        ])

        with self.assertRaises(ValueError):
            validate_transition_matrix(matrix)

    def test_the_regime_holds_until_the_interval_and_then_draws_from_its_row(self):
        matrix = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        book = RegimeBook(
            np.zeros((3, 1)),
            np.ones((3, 1)),
            num_paths=2,
            trigger_index=0,
            initial_states=np.array([1, 2]),
            transition=matrix,
            transition_interval=3,
            rng=_Draws([0.0, 0.0]),
        )

        book.observe()
        book.observe()
        self.assertEqual(book.states.tolist(), [1, 2])

        book.observe()
        self.assertEqual(book.states.tolist(), [3, 1])

    def test_zscore_is_the_closes_distance_from_its_average(self):
        prices = np.array([[1.0], [1.0], [1.0], [0.5]])
        average = prices.mean()
        sigma = prices.std()

        np.testing.assert_allclose(sma_zscore(prices, 4), [(0.5 - average) / sigma])
        np.testing.assert_allclose(sma_zscore(prices, 5), [0.0])
        np.testing.assert_allclose(sma_zscore(np.ones((4, 2)), 4), [0.0, 0.0])

    def test_one_standard_deviation_raises_bull_or_raises_churn_and_bear(self):
        matrix = np.array([
            [0.50, 0.20, 0.30],
            [0.40, 0.40, 0.20],
            [0.20, 0.10, 0.70],
        ])

        above, below = tilt_transition(matrix, np.array([1.0, -1.0]), per_std=0.10)

        np.testing.assert_allclose(above[0], [0.60, 0.16, 0.24])
        np.testing.assert_allclose(above[1], [0.50, 0.40 - (1 / 15), 0.20 - (1 / 30)])
        np.testing.assert_allclose(above[2], [0.30, 0.0875, 0.6125])
        np.testing.assert_allclose(below[0], [0.40, 0.24, 0.36])
        np.testing.assert_allclose(below[1], [0.30, 0.40 + (1 / 15), 0.20 + (1 / 30)])
        np.testing.assert_allclose(below[2], [0.10, 0.1125, 0.7875])
        np.testing.assert_allclose(above.sum(axis=1), 1.0)
        np.testing.assert_allclose(below.sum(axis=1), 1.0)

        pure_bull = tilt_transition(np.eye(3), np.array([-1.0]), per_std=0.20)[0]
        np.testing.assert_allclose(pure_bull[0], [0.80, 0.10, 0.10])
        np.testing.assert_allclose(pure_bull[1], [0.0, 1.0, 0.0])

    def test_the_tilt_stops_at_zero_and_at_three_standard_deviations(self):
        matrix = np.tile([0.40, 0.20, 0.40], (3, 1))

        wiped = tilt_transition(matrix, np.array([-10.0]), per_std=1.0)[0]
        capped = tilt_transition(matrix, np.array([100.0]), per_std=0.10)
        three = tilt_transition(matrix, np.array([3.0]), per_std=0.10)

        np.testing.assert_allclose(wiped[:, 0], 0.0)
        np.testing.assert_allclose(wiped[:, 1], 1 / 3)
        np.testing.assert_allclose(wiped[:, 2], 2 / 3)
        np.testing.assert_allclose(capped, three)

    def test_a_close_above_the_average_keeps_bull_that_the_base_draw_would_leave(self):
        matrix = np.tile([0.50, 0.00, 0.50], (3, 1))
        book = RegimeBook(
            np.zeros((3, 1)),
            np.ones((3, 1)),
            num_paths=1,
            trigger_index=0,
            initial_states=np.array([1]),
            transition=matrix,
            transition_interval=4,
            sma_window=4,
            sma_per_std=1.0,
            rng=_Draws([0.60]),
        )
        for growth in (0.5, 1.0, 1.0, 2.0):
            book.observe(np.array([growth]))

        self.assertEqual(book.states.tolist(), [1])

    def test_a_falling_trigger_moves_the_path_into_bear(self):
        matrix = np.tile([0.50, 0.00, 0.50], (3, 1))

        def arguments(window):
            return dict(
                S0=np.array([1.0]),
                drift=np.array([0.0]),
                volatility=np.array([0.0]),
                T=5.0,
                dt=1.0,
                num_paths=1,
                lower_threshold=np.array([0.0]),
                upper_threshold=np.array([2.0]),
                target_allocation=np.array([1.0]),
                regime_drift=np.array([[-2.0], [0.0], [0.0]]),
                regime_volatility=np.zeros((3, 1)),
                trigger_index=0,
                initial_states=np.array([1]),
                transition=matrix,
                transition_interval=4,
                regime_rng=_Draws([0.40]),
                sma_window=window,
                sma_per_std=1.0,
            )

        tilted, _lower, _upper = simulate_with_rebalance(**arguments(4))
        plain, _lower, _upper = simulate_with_rebalance(**arguments(100))

        np.testing.assert_allclose(tilted[5, 0, 0], tilted[4, 0, 0])
        np.testing.assert_allclose(plain[5, 0, 0], plain[4, 0, 0] * np.exp(-2.0))

    def test_rebalancing_the_book_does_not_move_the_trigger_average(self):
        matrix = np.tile([0.50, 0.00, 0.50], (3, 1))
        states = np.zeros((6, 1), dtype=int)
        simulate_with_rebalance(
            S0=np.array([1.0, 1.0]),
            drift=np.array([0.0, 0.0]),
            volatility=np.zeros(2),
            T=5.0,
            dt=1.0,
            num_paths=1,
            lower_threshold=np.array([0.40, 0.40]),
            upper_threshold=np.array([0.60, 0.60]),
            target_allocation=np.array([0.5, 0.5]),
            rebalance_interval=1,
            regime_drift=np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            regime_volatility=np.zeros((3, 2)),
            trigger_index=0,
            initial_states=np.array([1]),
            transition=matrix,
            transition_interval=4,
            regime_rng=_Draws([0.60]),
            regime_states=states,
            sma_window=4,
            sma_per_std=1.0,
        )

        self.assertEqual(int(states[5, 0]), 3)

    def test_sheet_round_trip_keeps_bull_churn_and_bear_in_state_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            workbook = openpyxl.Workbook()
            add_transition_sheet(workbook)
            workbook.save(path)

            matrix, interval, per_std, z_cap = load_transition_matrix(path)

        np.testing.assert_allclose(matrix, DEFAULT_TRANSITION)
        self.assertEqual(interval, DEFAULT_TRANSITION_INTERVAL)
        self.assertEqual(per_std, SMA_PROBABILITY_PER_STD)
        self.assertEqual(z_cap, SMA_Z_CAP)


class StateParameterEstimateTest(unittest.TestCase):
    def test_a_rising_market_is_bull_and_a_falling_market_is_bear(self):
        from gbm.regime import estimate_state_parameters

        rising = np.repeat(1.03, 18)
        falling = np.repeat(0.96, 18)
        flat = np.repeat(1.001, 18)
        growth = np.concatenate([[1.0], rising, falling, flat])
        index = pd.date_range("1996-01-31", periods=len(growth), freq="ME")
        market = pd.Series(100.0 * np.cumprod(growth), index=index)
        asset = pd.Series(50.0 * np.cumprod(growth), index=index)

        parameters = estimate_state_parameters(asset, market)

        self.assertGreater(parameters["State 1 Drift"], 0.20)
        self.assertLess(parameters["State 3 Drift"], -0.20)
        self.assertGreater(parameters["State 2 Drift"], parameters["State 3 Drift"])
        self.assertLess(parameters["State 2 Drift"], parameters["State 1 Drift"])
        for name in ("State 1 Volatility", "State 2 Volatility", "State 3 Volatility"):
            self.assertGreaterEqual(parameters[name], 0.0)
            self.assertTrue(np.isfinite(parameters[name]))

    def test_a_market_copy_keeps_the_scenario_row(self):
        from gbm.regime import scale_state_parameters

        historical = {
            "State 1 Drift": 0.40, "State 1 Volatility": 0.12,
            "State 2 Drift": 0.02, "State 2 Volatility": 0.18,
            "State 3 Drift": -0.90, "State 3 Volatility": 0.20,
        }
        scenario = {
            "State 1 Drift": 0.30, "State 1 Volatility": 0.14,
            "State 2 Drift": 0.05, "State 2 Volatility": 0.25,
            "State 3 Drift": -0.40, "State 3 Volatility": 0.40,
        }

        scaled = scale_state_parameters(historical, historical, scenario)

        for name, value in scenario.items():
            self.assertAlmostEqual(scaled[name], value)

    def test_csvyx_uses_the_small_value_scenario_and_fee(self):
        workbook = r"C:\Users\Evan\VirtualBox VMs\Sandbox\PortfolioManagement\portfolio_gbm_calculator.xlsx"
        behavior = pd.read_excel(workbook, sheet_name="Asset State Behaviors")
        behavior = behavior.set_index(behavior["Ticker"].astype(str).str.strip())
        expenses = pd.read_excel(workbook, sheet_name="Net Expense Ratios")
        expenses = expenses.set_index(expenses["Ticker"].astype(str).str.strip())
        row = behavior.loc["CSVYX"]

        self.assertAlmostEqual(row["State 1 Drift"], 0.286542, places=6)
        self.assertAlmostEqual(row["State 1 Volatility"], 0.253218, places=6)
        self.assertAlmostEqual(row["State 2 Drift"], 0.071359, places=6)
        self.assertAlmostEqual(row["State 2 Volatility"], 0.334109, places=6)
        self.assertAlmostEqual(row["State 3 Drift"], -0.412058, places=6)
        self.assertAlmostEqual(row["State 3 Volatility"], 0.726973, places=6)
        self.assertAlmostEqual(row["Momentum Lambda"], -0.310842, places=6)
        self.assertAlmostEqual(row["Momentum Center"], 0.103804, places=6)
        self.assertEqual(expenses.loc["CSVYX", "Fund / Asset Name"], "Columbia Small Cap Value Discovery Fund")
        self.assertAlmostEqual(expenses.loc["CSVYX", "Net Expense Ratio"], 0.0081)


class RegimeSimulationTest(unittest.TestCase):
    def test_sheet_parameters_follow_each_asset_and_cash_flight(self):
        frame = pd.DataFrame([
            ["TQQQ", 0.48, 0.35, -0.05, 0.50, -0.75, 0.90],
            ["DBMF", 0.04, 0.10, 0.065, 0.12, 0.12, 0.18],
            ["SGOV", 0.0525, 0.0, 0.0525, 0.0, 0.0525, 0.0],
        ], columns=[
            "Ticker",
            "State 1 Drift", "State 1 Volatility",
            "State 2 Drift", "State 2 Volatility",
            "State 3 Drift", "State 3 Volatility",
        ])

        drift, vol = align_regime_parameters(frame, ["TQQQ", "DBMF"], cash_flight=("SPMO", "DWAS", "TQQQ"))

        np.testing.assert_allclose(drift[:, 1], [0.04, 0.065, 0.12])
        np.testing.assert_allclose(vol[:, 1], [0.10, 0.12, 0.18])
        np.testing.assert_allclose(drift[:2, 0], [0.48, -0.05])
        np.testing.assert_allclose([drift[2, 0], vol[2, 0]], [0.0525, 0.0])

    def test_voo_sleeve_uses_the_spmo_behavior_row(self):
        frame = pd.DataFrame([
            ["SPMO", 0.22, 0.14, -0.01, 0.19, -0.28, 0.34],
            ["SGOV", 0.04, 0.0, 0.04, 0.0, 0.04, 0.0],
        ], columns=[
            "Ticker",
            "State 1 Drift", "State 1 Volatility",
            "State 2 Drift", "State 2 Volatility",
            "State 3 Drift", "State 3 Volatility",
        ])

        drift, _vol = align_regime_parameters(frame, ["VOO/VIIX"], cash_flight=())

        np.testing.assert_allclose(drift[:, 0], [0.22, -0.01, -0.28])

    def test_each_path_uses_its_own_regime_drift(self):
        drift = np.array([[0.50], [0.0], [-0.40]])
        vol = np.zeros((3, 1))
        states = np.array([1, 3])
        prices, _lower, _upper = simulate_with_rebalance(
            S0=np.array([1.0]),
            drift=np.array([0.0]),
            volatility=np.array([0.0]),
            T=1 / 252,
            dt=1 / 252,
            num_paths=2,
            lower_threshold=np.array([0.0]),
            upper_threshold=np.array([2.0]),
            target_allocation=np.array([1.0]),
            regime_drift=drift,
            regime_volatility=vol,
            trigger_index=0,
            initial_states=states,
        )

        growth = np.exp(drift[states - 1, 0] * (1 / 252))
        np.testing.assert_allclose(prices[1, :, 0], growth)
        self.assertGreater(prices[1, 0, 0], prices[1, 1, 0])

    def test_voo_volatility_rises_after_three_churn_months(self):
        self.assertEqual(churn_volatility_addon(np.array([0, 62, 63, 83, 84, 126])).tolist(), [0.0, 0.0, 0.1, 0.1, 0.2, 0.4])

        drift = np.zeros((3, 2))
        volatility = np.array([
            [0.20, 0.30],
            [0.11, 0.40],
            [0.50, 0.60],
        ])
        book = RegimeBook(drift, volatility, num_paths=2, trigger_index=0, initial_states=np.array([2, 1]))
        book.days_in_churn = np.array([84, 84])

        _drift, vol = book.parameters()

        self.assertAlmostEqual(vol[0, 0], 0.31)
        self.assertAlmostEqual(vol[0, 1], 0.40)
        self.assertAlmostEqual(vol[1, 0], 0.20)

    def test_three_flat_months_do_not_raise_voo_volatility_and_the_fourth_does(self):
        drift = np.zeros((3, 1))
        volatility = np.array([[0.2], [0.11], [0.5]])
        book = RegimeBook(
            drift,
            volatility,
            num_paths=1,
            trigger_index=0,
            transition=np.eye(3),
        )
        for _ in range(62):
            book.observe(np.array([1.0]))
        self.assertAlmostEqual(book.parameters()[1][0, 0], 0.11)

        book.observe(np.array([1.0]))
        self.assertAlmostEqual(book.parameters()[1][0, 0], 0.21)

    def test_the_next_step_uses_the_regime_drawn_from_the_transition_matrix(self):
        drift = np.array([[0.50], [0.0], [-0.40]])
        vol = np.zeros((3, 1))
        matrix = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        prices, _lower, _upper = simulate_with_rebalance(
            S0=np.array([1.0]),
            drift=np.array([0.0]),
            volatility=np.array([0.0]),
            T=2 / 252,
            dt=1 / 252,
            num_paths=1,
            lower_threshold=np.array([0.0]),
            upper_threshold=np.array([2.0]),
            target_allocation=np.array([1.0]),
            regime_drift=drift,
            regime_volatility=vol,
            trigger_index=0,
            initial_states=np.array([1]),
            transition=matrix,
            transition_interval=1,
            regime_rng=_Draws([0.5, 0.5]),
        )

        dt = 1 / 252
        first = np.exp(0.50 * dt)
        second = first * np.exp(-0.40 * dt)
        np.testing.assert_allclose(prices[:, 0, 0], [1.0, first, second])


class TransitionSheetTest(unittest.TestCase):
    def test_workbook_transition_is_what_the_run_loads(self):
        workbook = openpyxl.load_workbook(WORKBOOK)
        sheet = workbook[TRANSITION_SHEET]
        table = sheet.tables["TransitionMatrix"]
        used = np.array([
            [0.88, 0.07, 0.05],
            [0.15, 0.70, 0.15],
            [0.20, 0.10, 0.70],
        ])

        self.assertEqual(table.ref, "A1:E4")
        self.assertEqual(
            [cell.value for cell in sheet[1]],
            ["From", "Bull", "Churn", "Bear", "Note", "Expected Duration (transitions)"],
        )
        for row, note in zip(range(2, 5), DEFAULT_TRANSITION_NOTES):
            self.assertEqual(sheet.cell(row, 5).value, note)
            self.assertEqual(sheet.cell(row, 2).number_format, "0%")
        np.testing.assert_allclose(
            [[sheet.cell(row, column).value for column in range(2, 5)] for row in range(2, 5)],
            used,
        )
        self.assertEqual(sheet["A6"].value, TRANSITION_INTERVAL_LABEL)
        self.assertEqual(sheet["B6"].value, DEFAULT_TRANSITION_INTERVAL)
        self.assertEqual(sheet["A10"].value, TRANSITION_PER_STD_LABEL)
        self.assertEqual(sheet["B10"].value, 0.08)
        self.assertEqual(sheet["A11"].value, TRANSITION_Z_CAP_LABEL)
        self.assertEqual(sheet["B11"].value, SMA_Z_CAP)
        matrix, interval, per_std, z_cap = load_transition_matrix(WORKBOOK)
        np.testing.assert_allclose(matrix, used)
        self.assertEqual(interval, DEFAULT_TRANSITION_INTERVAL)
        self.assertEqual(per_std, 0.08)
        self.assertEqual(z_cap, SMA_Z_CAP)

    def test_sheet_values_replace_the_standard_deviation_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            workbook = openpyxl.Workbook()
            add_transition_sheet(workbook)
            sheet = workbook[TRANSITION_SHEET]
            sheet["B10"] = 0.25
            sheet["B11"] = 2
            workbook.save(path)

            _matrix, _interval, per_std, z_cap = load_transition_matrix(path)

        self.assertEqual(per_std, 0.25)
        self.assertEqual(z_cap, 2.0)


if __name__ == "__main__":
    unittest.main()
