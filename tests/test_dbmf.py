import unittest

import numpy as np
import pandas as pd

from gbm.dbmf import (
    CHURN_EQUITY_CORRELATION,
    EXPANSION_EQUITY_CORRELATION,
    LIQUIDATION_EQUITY_CORRELATION,
)
from gbm.gbm_step import (
    ASSET_CORRELATION,
    CORRELATION_TICKERS,
    regime_correlation_matrices,
    verify_and_generate_cholesky,
)
from gbm.momentum import align_momentum_drift


class ManagedFuturesCorrelationTest(unittest.TestCase):
    """DBMF diversifies against equities. The crash blend must not pull it in."""

    LABELS = ["VOO", "DBMF", "KMLM", "GLD", "SGOV"]
    BASELINE = np.array([
        [1.00, 0.17, -0.17, 0.09, -0.02],
        [0.17, 1.00, 0.52, 0.17, -0.03],
        [-0.17, 0.52, 1.00, 0.10, 0.00],
        [0.09, 0.17, 0.10, 1.00, 0.01],
        [-0.02, -0.03, 0.00, 0.01, 1.00],
    ])

    def test_ordinary_months_stay_near_the_live_monthly_correlation(self):
        expansion, churn, _liquidation = regime_correlation_matrices(self.BASELINE, self.LABELS)

        np.testing.assert_allclose(expansion[1, 0], EXPANSION_EQUITY_CORRELATION)
        np.testing.assert_allclose(churn[1, 0], CHURN_EQUITY_CORRELATION)
        np.testing.assert_allclose(expansion[1, 4], -0.03)
        np.testing.assert_allclose(expansion[1, 2], 0.52)
        np.testing.assert_allclose(expansion[1, 3], 0.17)
        self.assertLess(expansion[1, 0], 0.0)
        self.assertGreater(expansion[1, 0], -0.20)

    def test_a_crash_pays_dbmf_when_equities_fall_and_leaves_the_peers_alone(self):
        _expansion, _churn, liquidation = regime_correlation_matrices(self.BASELINE, self.LABELS)

        np.testing.assert_allclose(liquidation[1, 0], LIQUIDATION_EQUITY_CORRELATION)
        np.testing.assert_allclose(liquidation[1, 4], -0.03)
        np.testing.assert_allclose(liquidation[1, 2], 0.52)
        np.testing.assert_allclose(liquidation[1, 3], 0.17)
        self.assertLess(liquidation[1, 0], -0.25)
        self.assertGreater(liquidation[0, 4], 0.50)
        factor = verify_and_generate_cholesky(liquidation)
        shock = np.array([-2.0, 0.0, 0.0, 0.0, 0.0])
        self.assertGreater(float((factor @ shock)[1]), 0.0)

    def test_the_saved_universe_keeps_the_same_diversifier(self):
        recovered = ASSET_CORRELATION @ ASSET_CORRELATION.T
        labels = list(CORRELATION_TICKERS)
        expansion, churn, liquidation = regime_correlation_matrices(recovered, labels)
        dbmf = labels.index("DBMF")
        voo = labels.index("VOO")
        kmlm = labels.index("KMLM")
        gld = labels.index("GLD")
        sgov = labels.index("SGOV")

        self.assertLess(expansion[dbmf, voo], 0.0)
        self.assertGreater(expansion[dbmf, voo], -0.20)
        self.assertLess(liquidation[dbmf, voo], -0.25)
        self.assertGreater(liquidation[dbmf, voo], -0.55)
        self.assertGreater(churn[dbmf, voo], expansion[dbmf, voo])
        self.assertLess(churn[dbmf, voo], 0.0)
        np.testing.assert_allclose(liquidation[dbmf, kmlm], recovered[dbmf, kmlm], atol=0.12)
        np.testing.assert_allclose(liquidation[dbmf, gld], recovered[dbmf, gld], atol=0.08)
        np.testing.assert_allclose(liquidation[dbmf, sgov], recovered[dbmf, sgov], atol=0.08)
        self.assertGreater(liquidation[dbmf, kmlm], 0.40)
        for matrix in (expansion, churn, liquidation):
            self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0.0)
            np.testing.assert_allclose(np.diag(matrix), np.ones(len(labels)))

    def test_a_book_without_dbmf_keeps_the_flat_crash_blend(self):
        baseline = np.array([[1.0, 0.40], [0.40, 1.0]])

        expansion, _churn, liquidation = regime_correlation_matrices(baseline, ["VOO", "SGOV"])

        np.testing.assert_allclose(expansion, baseline)
        self.assertGreater(liquidation[0, 1], 0.50)


class ManagedFuturesMomentumTest(unittest.TestCase):
    def test_a_cached_dbmf_slope_is_kept_until_the_behavior_sheet_replaces_it(self):
        frame = pd.DataFrame({
            "Ticker": ["VOO", "DBMF", "KMLM"],
            "Lambda": [-0.33, -0.52, -0.16],
            "Center": [0.14, 0.066, 0.04],
        })

        lambda_mom, center = align_momentum_drift(frame, ["VOO", "DBMF", "SGOV"])

        np.testing.assert_allclose(lambda_mom, [-0.33, -0.52, 0.0])
        np.testing.assert_allclose(center, [0.14, 0.066, 0.0])
