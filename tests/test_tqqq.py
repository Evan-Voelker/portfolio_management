import unittest
from unittest import mock

import numpy as np
import pandas as pd

from gbm.gbm_step import gbm_step_vectorized_3d
from gbm.regime import estimate_state_parameters, scale_state_parameters
from gbm.tqqq import TQQQ_EXPENSE, TQQQ_LEVERAGE, leveraged_drift


class LeveragedResetTest(unittest.TestCase):
    def test_a_session_is_three_times_the_index_minus_financing_and_the_fee(self):
        from gbm.asset_simulation_matrix import leveraged_reset_level

        index = pd.to_datetime(["2020-01-03", "2020-01-06"])
        closes = pd.Series([100.0, 110.0], index=index)
        yields = pd.Series([3.65, 3.65], index=index)

        level = leveraged_reset_level(closes, yields, leverage=3.0, expense=0.0086)

        financing = 2.0 * 0.0365 * 3.0 / 365.0
        expense = 0.0086 * 3.0 / 365.0
        self.assertEqual(level.name, "TQQQ")
        np.testing.assert_allclose(level.iloc[0], 1.0)
        np.testing.assert_allclose(level.iloc[1], 1.0 + 3.0 * 0.10 - financing - expense)

    def test_a_wipeout_session_stays_positive(self):
        from gbm.asset_simulation_matrix import leveraged_reset_level

        index = pd.to_datetime(["2020-01-06", "2020-01-07"])
        closes = pd.Series([100.0, 50.0], index=index)
        yields = pd.Series([0.0, 0.0], index=index)

        level = leveraged_reset_level(closes, yields, leverage=3.0, expense=0.0)

        self.assertGreater(level.iloc[1], 0.0)
        self.assertLess(level.iloc[1], 1e-4)


class TqqqBackfillTest(unittest.TestCase):
    def test_synthetic_history_is_scaled_onto_the_first_tqqq_print(self):
        from gbm.asset_simulation_matrix import combine_tqqq

        index = pd.to_datetime(["2010-02-10", "2010-02-11", "2010-02-12"])
        etf = pd.Series([np.nan, 40.0, 42.0], index=index)
        synthetic = pd.Series([10.0, 12.0, 11.0], index=index)

        combined = combine_tqqq(etf, synthetic)

        np.testing.assert_allclose(combined.iloc[0], 10.0 * 40.0 / 12.0)
        np.testing.assert_allclose(combined.iloc[1], 40.0)
        np.testing.assert_allclose(combined.iloc[2], 42.0)
        np.testing.assert_allclose(combined.iloc[1] / combined.iloc[0] - 1.0, 12.0 / 10.0 - 1.0)
        self.assertEqual(combined.name, "TQQQ")


class LeveragedRegimeTest(unittest.TestCase):
    def test_drift_borrows_at_the_financing_rate_and_volatility_scales(self):
        drift, volatility = leveraged_drift(0.10, 0.20, financing=0.04, expense=TQQQ_EXPENSE)

        self.assertAlmostEqual(drift, TQQQ_LEVERAGE * 0.10 - (TQQQ_LEVERAGE - 1.0) * 0.04 - TQQQ_EXPENSE)
        self.assertAlmostEqual(volatility, TQQQ_LEVERAGE * 0.20)

    def test_daily_reset_decay_is_the_step_half_variance(self):
        index_drift = np.array([0.30, 0.05, -0.40])
        index_vol = np.array([0.14, 0.25, 0.40])
        drift, volatility = leveraged_drift(index_drift, index_vol, financing=0.04, expense=TQQQ_EXPENSE)
        log_growth = drift - 0.5 * volatility**2
        index_log = index_drift - 0.5 * index_vol**2
        decay = 0.5 * TQQQ_LEVERAGE * (TQQQ_LEVERAGE - 1.0) * index_vol**2
        financing = (TQQQ_LEVERAGE - 1.0) * 0.04 + TQQQ_EXPENSE

        np.testing.assert_allclose(log_growth, TQQQ_LEVERAGE * index_log - decay - financing)

    def test_a_three_times_history_scales_onto_three_times_the_scenario(self):
        index = pd.date_range("1996-01-31", periods=55, freq="ME")
        noise = np.array([
            0.00, 0.01, -0.01, 0.02, -0.005, 0.015, -0.02, 0.005, 0.01, -0.015, 0.02,
        ])
        bull = 1.05 + np.resize(noise, 18)
        bear = 0.94 + np.resize(noise, 18) * 0.25
        churn = 1.004 + np.resize(noise, 18) * 0.05
        market_growth = np.concatenate([[1.0], bull, bear, churn])
        market = pd.Series(100.0 * np.cumprod(market_growth), index=index)
        asset_growth = 1.0 + TQQQ_LEVERAGE * (market_growth - 1.0)
        asset_growth[0] = 1.0
        asset = pd.Series(50.0 * np.cumprod(asset_growth), index=index)
        scenario = {
            "State 1 Drift": 0.30, "State 1 Volatility": 0.14,
            "State 2 Drift": 0.05, "State 2 Volatility": 0.25,
            "State 3 Drift": -0.40, "State 3 Volatility": 0.40,
        }

        scaled = scale_state_parameters(
            estimate_state_parameters(asset, market),
            estimate_state_parameters(market, market),
            scenario,
        )

        for name in ("State 1 Drift", "State 2 Drift", "State 3 Drift"):
            self.assertAlmostEqual(scaled[name], TQQQ_LEVERAGE * scenario[name])
        for name in ("State 1 Volatility", "State 2 Volatility", "State 3 Volatility"):
            self.assertGreater(scaled[name], scenario[name])

    def test_one_gbm_step_compounds_each_regime_at_the_leveraged_drift(self):
        index_drift = np.array([0.30, 0.05, -0.40])
        index_vol = np.array([0.14, 0.25, 0.40])
        drift, volatility = leveraged_drift(index_drift, index_vol, financing=0.04, expense=0.0)
        dt = 1.0 / 252.0
        prices = np.ones((3, 1))
        step_drift = drift.reshape(3, 1)
        step_vol = volatility.reshape(3, 1)

        with mock.patch("gbm.gbm_step.np.random.standard_normal", return_value=np.zeros((3, 1))):
            stepped = gbm_step_vectorized_3d(prices, step_drift, step_vol, dt)

        expected = np.exp((step_drift - 0.5 * step_vol**2) * dt)
        np.testing.assert_allclose(stepped, expected)
        self.assertGreater(stepped[0, 0], stepped[1, 0])
        self.assertGreater(stepped[1, 0], stepped[2, 0])

    def test_leverage_does_not_change_correlation(self):
        rng = np.random.default_rng(7)
        market = rng.normal(0.0004, 0.01, 500)
        other = 0.6 * market + rng.normal(0.0, 0.008, 500)
        leveraged = TQQQ_LEVERAGE * market - 0.0002

        plain = np.corrcoef(market, other)[0, 1]
        scaled = np.corrcoef(leveraged, other)[0, 1]

        np.testing.assert_allclose(scaled, plain, atol=1e-12)


class TqqqWorkbookTest(unittest.TestCase):
    def test_behavior_row_is_the_nasdaq_history_scaled_into_the_voo_scenario(self):
        from gbm.asset_simulation_matrix import WORKBOOK_PATH

        frame = pd.read_excel(WORKBOOK_PATH, sheet_name="Asset State Behaviors")
        frame = frame.set_index(frame["Ticker"].astype(str).str.strip())
        row = frame.loc["TQQQ"]

        self.assertAlmostEqual(row["State 1 Drift"], 1.033867, places=6)
        self.assertAlmostEqual(row["State 1 Volatility"], 0.630764, places=6)
        self.assertAlmostEqual(row["State 2 Drift"], 0.291476, places=6)
        self.assertAlmostEqual(row["State 2 Volatility"], 1.136571, places=6)
        self.assertAlmostEqual(row["State 3 Drift"], -1.554313, places=6)
        self.assertAlmostEqual(row["State 3 Volatility"], 1.351085, places=6)
        self.assertAlmostEqual(row["Momentum Lambda"], -0.016059, places=6)
        self.assertAlmostEqual(row["Momentum Center"], 0.500732, places=6)

        expenses = pd.read_excel(WORKBOOK_PATH, sheet_name="Net Expense Ratios")
        expenses = expenses.set_index(expenses["Ticker"].astype(str).str.strip())
        self.assertEqual(expenses.loc["TQQQ", "Fund / Asset Name"], "ProShares UltraPro QQQ (3x Leverage)")
        self.assertAlmostEqual(expenses.loc["TQQQ", "Net Expense Ratio"], 0.0086)

    def test_extended_history_keeps_tqqq_with_equities_rather_than_cash(self):
        from gbm.asset_simulation_matrix import CHOLESKY_PATH

        factor = pd.read_csv(CHOLESKY_PATH, index_col=0)
        recovered = factor.to_numpy() @ factor.to_numpy().T
        labels = [str(label).strip() for label in factor.index]
        tqqq = labels.index("TQQQ")
        equity = recovered[tqqq, labels.index("VOO")]
        cash = recovered[tqqq, labels.index("SGOV")]

        self.assertGreater(equity, 0.85)
        self.assertLess(abs(cash), 0.05)
