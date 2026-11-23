import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gbm.gbm_step import (
    ASSET_CORRELATION,
    CORRELATION_TICKERS,
    apply_cholesky,
    correlation_for_labels,
    gbm_step_vectorized_3d,
    nearest_correlation,
    regime_cholesky_factors,
    regime_correlation_matrices,
    simulation_cholesky,
    simulate_with_rebalance,
    verify_and_generate_cholesky,
)


class CholeskyCorrelationTest(unittest.TestCase):
    def test_asset_correlation_is_the_saved_cholesky_factor(self):
        self.assertEqual(ASSET_CORRELATION.shape, (len(CORRELATION_TICKERS), len(CORRELATION_TICKERS)))
        np.testing.assert_allclose(np.triu(ASSET_CORRELATION, k=1), np.zeros_like(ASSET_CORRELATION), atol=1e-8)
        recovered = ASSET_CORRELATION @ ASSET_CORRELATION.T
        np.testing.assert_allclose(np.diag(recovered), np.ones(len(CORRELATION_TICKERS)), atol=1e-6)
        np.testing.assert_allclose(recovered, recovered.T, atol=1e-6)

    def test_correlation_for_labels_follows_the_sheet_and_matches_voo(self):
        labels = ["GLD", "VOO/VIIX", "KMLM"]
        aligned = correlation_for_labels(labels)
        recovered = ASSET_CORRELATION @ ASSET_CORRELATION.T
        source = [CORRELATION_TICKERS.index(ticker) for ticker in ("GLD", "VOO", "KMLM")]
        expected = recovered[np.ix_(source, source)]

        np.testing.assert_allclose(aligned, expected)
        np.testing.assert_allclose(np.diag(aligned), np.ones(3))

    def test_sheet_order_factor_matches_the_saved_cholesky(self):
        factor = verify_and_generate_cholesky(correlation_for_labels(CORRELATION_TICKERS))

        np.testing.assert_allclose(factor, ASSET_CORRELATION, atol=1e-6)

    def test_step_mixes_independent_shocks_with_the_cholesky_factor(self):
        prices = np.array([[100.0, 80.0, 40.0]])
        drift = np.array([[0.10, 0.07, 0.22]])
        volatility = np.array([[0.15, 0.10, 0.50]])
        dt = 1 / 252
        independent = np.array([[0.4, -1.1, 0.2]])
        correlation = np.array([
            [1.00, 0.82, -0.15],
            [0.82, 1.00, -0.10],
            [-0.15, -0.10, 1.00],
        ])
        cholesky = verify_and_generate_cholesky(correlation)
        correlated = independent @ cholesky.T
        expected = prices * np.exp(
            (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * correlated
        )

        with patch("gbm.gbm_step.np.random.standard_normal", return_value=independent):
            stepped = gbm_step_vectorized_3d(prices, drift, volatility, dt, cholesky=cholesky)

        np.testing.assert_allclose(stepped, expected)

    def test_one_step_return_correlation_tracks_the_matrix(self):
        cholesky = ASSET_CORRELATION
        correlation = cholesky @ cholesky.T
        num_paths = 20000
        num_assets = cholesky.shape[0]
        prices = np.ones((num_paths, num_assets))
        drift = np.zeros((num_paths, num_assets))
        volatility = np.ones((num_paths, num_assets))
        dt = 1 / 252
        state = np.random.get_state()
        np.random.seed(7)
        try:
            stepped = gbm_step_vectorized_3d(prices, drift, volatility, dt, cholesky=cholesky)
        finally:
            np.random.set_state(state)

        log_returns = np.log(stepped / prices)
        sample = np.corrcoef(log_returns, rowvar=False)
        np.testing.assert_allclose(sample, correlation, atol=0.05)

    def test_a_sheet_factor_is_reordered_into_the_label_order(self):
        labels = ["VOO", "XLK", "BTC"]
        tickers = ["XLK", "BTC", "VOO"]
        factor = np.array([
            [1.0, 0.0, 0.0],
            [0.4, 0.91651514, 0.0],
            [0.2, 0.1, 0.97467943],
        ])

        aligned = correlation_for_labels(labels, tickers=tickers, factor=factor)

        recovered = factor @ factor.T
        order = [tickers.index(label) for label in labels]
        np.testing.assert_allclose(aligned, recovered[np.ix_(order, order)])

    def test_the_default_sheet_keeps_the_saved_factor(self):
        with patch("gbm.gbm_step.load_asset_cholesky") as loader:
            matrix = simulation_cholesky(["GLD", "VOO/VIIX"], "Asset Parameters")

        loader.assert_not_called()
        expected = verify_and_generate_cholesky(correlation_for_labels(["GLD", "VOO/VIIX"]))
        np.testing.assert_allclose(matrix, expected)

    def test_another_sheet_loads_the_factor_for_that_sheet(self):
        labels = ["XLK", "BTC"]
        factor = pd.DataFrame(np.eye(2), index=labels, columns=labels)
        with patch("gbm.gbm_step.load_asset_cholesky", return_value=factor) as loader:
            matrix = simulation_cholesky(labels, "Sector Momentum Playbook")

        self.assertEqual(loader.call_args.kwargs["sheet_name"], "Sector Momentum Playbook")
        np.testing.assert_allclose(matrix, np.eye(2))


class RegimeCholeskyTest(unittest.TestCase):
    def test_expansion_churn_and_liquidation_follow_the_baseline_recipes(self):
        baseline = np.array([
            [1.0, 0.40, -0.20],
            [0.40, 1.0, 0.10],
            [-0.20, 0.10, 1.0],
        ])

        expansion, churn, liquidation = regime_correlation_matrices(baseline)

        expected_churn = baseline * 0.75
        np.fill_diagonal(expected_churn, 1.0)
        expected_liquidation = 0.3 * baseline + 0.7 * np.full((3, 3), 0.85)
        np.fill_diagonal(expected_liquidation, 1.0)
        np.testing.assert_allclose(expansion, baseline)
        np.testing.assert_allclose(churn, expected_churn)
        np.testing.assert_allclose(liquidation, expected_liquidation)
        self.assertGreater(liquidation[0, 2], 0.5)
        self.assertLess(churn[0, 1], baseline[0, 1])

    def test_factors_are_lower_triangular_and_recover_each_regime(self):
        baseline = np.array([
            [1.0, 0.40, -0.20],
            [0.40, 1.0, 0.10],
            [-0.20, 0.10, 1.0],
        ])

        factors = regime_cholesky_factors(baseline)

        self.assertEqual(factors.shape, (3, 3, 3))
        for factor, correlation in zip(factors, regime_correlation_matrices(baseline)):
            np.testing.assert_allclose(np.triu(factor, k=1), np.zeros((3, 3)), atol=1e-12)
            np.testing.assert_allclose(factor @ factor.T, correlation, atol=1e-8)
            self.assertGreater(np.linalg.eigvalsh(correlation).min(), 0.0)

    def test_nearest_correlation_repairs_an_indefinite_matrix(self):
        broken = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])

        repaired = nearest_correlation(broken)

        np.testing.assert_allclose(np.diag(repaired), np.ones(3))
        np.testing.assert_allclose(repaired, repaired.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(repaired).min(), -1e-12)
        np.linalg.cholesky(repaired)

    def test_a_path_step_uses_the_factor_for_its_own_regime(self):
        baseline = np.array([[1.0, 0.20], [0.20, 1.0]])
        factor = np.linalg.cholesky(baseline)
        factors = regime_cholesky_factors(baseline)
        shocks = np.array([
            [0.40, -0.80],
            [0.40, -0.80],
            [0.40, -0.80],
        ])
        states = np.array([1, 2, 3])
        dt = 1 / 252
        with patch("gbm.gbm_step.np.random.standard_normal", return_value=shocks):
            prices, lower_counts, upper_counts = simulate_with_rebalance(
                S0=np.array([1.0, 1.0]),
                drift=np.zeros(2),
                volatility=np.ones(2),
                T=dt,
                dt=dt,
                num_paths=3,
                lower_threshold=np.array([-1.0, -1.0]),
                upper_threshold=np.array([2.0, 2.0]),
                target_allocation=np.array([0.5, 0.5]),
                cholesky=factor,
                regime_drift=np.zeros((3, 2)),
                regime_volatility=np.ones((3, 2)),
                trigger_index=0,
                initial_states=states,
            )

        correlated = np.stack([shocks[path] @ factors[state - 1].T for path, state in enumerate(states)])
        growth = np.exp(-0.5 * dt + np.sqrt(dt) * correlated)
        np.testing.assert_allclose(prices[1], growth)
        np.testing.assert_array_equal(lower_counts, np.zeros((3, 2)))
        np.testing.assert_array_equal(upper_counts, np.zeros((3, 2)))
        self.assertFalse(np.allclose(prices[1, 0], prices[1, 2]))

    def test_regime_states_use_the_matching_factor_without_a_per_path_stack(self):
        shocks = np.array([
            [0.2, -0.4],
            [0.2, -0.4],
            [0.5, 0.1],
            [-0.3, 0.7],
        ])
        factors = regime_cholesky_factors(np.array([[1.0, 0.2], [0.2, 1.0]]))
        states = np.array([1, 3, 2, 1])

        np.testing.assert_allclose(
            apply_cholesky(shocks, factors, states),
            apply_cholesky(shocks, factors[states - 1]),
        )

    def test_per_path_factors_mix_each_paths_shock_on_its_own(self):
        shocks = np.array([[0.4, -1.1], [0.4, -1.1]])
        factors = np.stack([
            np.linalg.cholesky(np.array([[1.0, 0.2], [0.2, 1.0]])),
            np.linalg.cholesky(np.array([[1.0, 0.9], [0.9, 1.0]])),
        ])
        prices = np.ones((2, 2))
        drift = np.zeros((2, 2))
        volatility = np.ones((2, 2))
        dt = 1 / 252
        correlated = np.einsum("pi,pji->pj", shocks, factors)
        expected = prices * np.exp(-0.5 * dt + np.sqrt(dt) * correlated)

        with patch("gbm.gbm_step.np.random.standard_normal", return_value=shocks):
            stepped = gbm_step_vectorized_3d(prices, drift, volatility, dt, cholesky=factors)

        np.testing.assert_allclose(stepped, expected)
        self.assertFalse(np.allclose(stepped[0], stepped[1]))


if __name__ == "__main__":
    unittest.main()
