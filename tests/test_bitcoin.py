import unittest

import numpy as np
import pandas as pd

from gbm.asset_simulation_matrix import combine_bitcoin
from gbm.bitcoin import (
    MATURE_BTC_STRESS_BLEND,
    MATURE_BTC_STRESS_CORRELATION,
    median_log_growth,
    stationary_weights,
)
from gbm.gbm_step import LIQUIDATION_BLEND, LIQUIDATION_CORRELATION, regime_correlation_matrices
from gbm.momentum import align_momentum_drift
from gbm.regime import DEFAULT_TRANSITION, align_regime_parameters


class MatureBitcoinRegimeTest(unittest.TestCase):
    def test_a_behavior_row_is_the_bitcoin_drift(self):
        frame = pd.DataFrame([
            ["BTC", 0.55, 0.50, 0.05, 0.75, -0.50, 1.10],
            ["VOO", 0.22, 0.14, -0.01, 0.19, -0.28, 0.34],
            ["SGOV", 0.04, 0.0, 0.04, 0.0, 0.04, 0.0],
        ], columns=[
            "Ticker",
            "State 1 Drift", "State 1 Volatility",
            "State 2 Drift", "State 2 Volatility",
            "State 3 Drift", "State 3 Volatility",
        ])

        drift, volatility = align_regime_parameters(frame, ["IBIT", "VOO"], cash_flight=())

        np.testing.assert_allclose(drift[:, 0], [0.55, 0.05, -0.50])
        np.testing.assert_allclose(volatility[:, 0], [0.50, 0.75, 1.10])
        np.testing.assert_allclose(drift[:, 1], [0.22, -0.01, -0.28])

    def test_the_equity_regime_mix_compounds_near_the_published_decade(self):
        # Bull row of the Transition Matrix sheet. Bear months are a little
        # more common than the code default, and that is the mix the run uses.
        workbook = np.array([
            [0.88, 0.07, 0.05],
            [0.15, 0.70, 0.15],
            [0.20, 0.10, 0.70],
        ])

        cagr = float(np.exp(median_log_growth(stationary_weights(workbook))) - 1.0)
        code_default = float(np.exp(median_log_growth(stationary_weights(DEFAULT_TRANSITION))) - 1.0)

        self.assertGreater(cagr, 0.25)
        self.assertLess(cagr, 0.29)
        self.assertLess(code_default, 0.32)


class MatureBitcoinMomentumTest(unittest.TestCase):
    def test_a_bitcoin_cache_row_is_kept(self):
        frame = pd.DataFrame(
            {"Ticker": ["VOO", "BTC"], "Lambda": [0.20, -0.15], "Center": [0.11, 1.38]}
        )

        lambda_mom, center = align_momentum_drift(frame, ["VOO", "SGOV", "IBIT"])

        np.testing.assert_allclose(lambda_mom, [0.20, 0.0, -0.15])
        np.testing.assert_allclose(center, [0.11, 0.0, 1.38])


class MatureBitcoinCorrelationTest(unittest.TestCase):
    def test_liquidation_pulls_btc_toward_risk_assets_and_leaves_cash_alone(self):
        baseline = np.array([
            [1.00, 0.20, -0.02],
            [0.20, 1.00, -0.01],
            [-0.02, -0.01, 1.00],
        ])
        labels = ["VOO", "BTC", "SGOV"]

        _expansion, _churn, liquidation = regime_correlation_matrices(baseline, labels)

        equity_crash = (1.0 - LIQUIDATION_BLEND) * -0.02 + LIQUIDATION_BLEND * LIQUIDATION_CORRELATION
        btc_equity = (1.0 - MATURE_BTC_STRESS_BLEND) * 0.20 + MATURE_BTC_STRESS_BLEND * MATURE_BTC_STRESS_CORRELATION
        np.testing.assert_allclose(liquidation[0, 2], equity_crash)
        np.testing.assert_allclose(liquidation[1, 0], btc_equity)
        np.testing.assert_allclose(liquidation[1, 2], -0.01)
        np.testing.assert_allclose(liquidation, liquidation.T)
        np.testing.assert_allclose(np.diag(liquidation), np.ones(3))
        self.assertLess(liquidation[1, 0], equity_crash)
        self.assertGreater(liquidation[1, 0], 0.20)
        self.assertLess(liquidation[1, 0], 0.50)


class BitcoinHistoryTest(unittest.TestCase):
    def test_spot_history_is_scaled_onto_the_first_etf_print(self):
        index = pd.to_datetime(["2024-01-01", "2024-01-02"])
        etf = pd.Series([np.nan, 20.0], index=index)
        spot = pd.Series([10.0, 99.0], index=index)

        combined = combine_bitcoin(etf, spot)

        np.testing.assert_allclose(combined.to_numpy(), [10.0 * 20.0 / 99.0, 20.0])
        self.assertEqual(combined.name, "BTC")
        join_return = combined.iloc[1] / combined.iloc[0] - 1.0
        spot_return = 99.0 / 10.0 - 1.0
        np.testing.assert_allclose(join_return, spot_return)
