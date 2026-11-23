import unittest

import numpy as np
import pandas as pd

from gbm.asset_simulation_matrix import load_asset_cholesky
from gbm.gbm_step import load_simulation_assumptions, regime_correlation_matrices

WORKBOOK = r"C:\Users\Evan\VirtualBox VMs\Sandbox\PortfolioManagement\portfolio_gbm_calculator.xlsx"
SHEET = "Sector Momentum Playbook"

SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
PLAYBOOK = SECTORS + ["VXUS", "BTC", "DBMF"]
# VOO's sector mix in October 2026. The eleven funds are the S&P 500, so this
# is the book that has to land on VOO's drift and volatility.
SECTOR_WEIGHTS = {
    "XLC": 0.0995,
    "XLY": 0.0875,
    "XLP": 0.0433,
    "XLE": 0.0340,
    "XLF": 0.1114,
    "XLV": 0.0926,
    "XLI": 0.0755,
    "XLB": 0.0156,
    "XLRE": 0.0170,
    "XLK": 0.4048,
    "XLU": 0.0188,
}


class SectorPlaybookSheetTest(unittest.TestCase):
    def test_asset_parameters_has_an_optional_blank_bucket_column(self):
        frame = pd.read_excel(WORKBOOK, sheet_name="Asset Parameters")

        self.assertIn("Bucket", frame.columns)
        blank = frame["Bucket"].isna() | frame["Bucket"].astype(str).str.strip().isin(["", "nan"])
        self.assertTrue(blank.all())

    def test_sector_playbook_lists_the_eleven_sectors_plus_vxus_btc_and_dbmf(self):
        frame = pd.read_excel(WORKBOOK, sheet_name=SHEET)

        self.assertIn("Bucket", frame.columns)
        self.assertIn("Asset Type", frame.columns)
        tickers = [str(ticker).strip() for ticker in frame["Ticker"]]
        for ticker in PLAYBOOK:
            self.assertIn(ticker, tickers)
        self.assertIn("SGOV", tickers)
        self.assertIn("VOO", tickers)

        indexed = frame.set_index(frame["Ticker"].map(lambda ticker: str(ticker).strip()))
        for ticker in SECTORS + ["VXUS", "SGOV"]:
            self.assertEqual(str(indexed.loc[ticker, "Asset Type"]).strip().lower(), "momentum")
        self.assertEqual(str(indexed.loc["BTC", "Asset Type"]).strip().lower(), "gated")
        self.assertIn(str(indexed.loc["DBMF", "Asset Type"]).strip().lower(), ("alternate", "alternative"))

    def test_sector_names_have_behavior_and_expense_rows(self):
        behaviors = pd.read_excel(WORKBOOK, sheet_name="Asset State Behaviors")
        expenses = pd.read_excel(WORKBOOK, sheet_name="Net Expense Ratios")
        behavior_names = set(behaviors["Ticker"].astype(str).str.strip())
        expense_names = set(expenses["Ticker"].astype(str).str.strip())

        for ticker in SECTORS:
            self.assertIn(ticker, behavior_names)
            self.assertIn(ticker, expense_names)

    def test_cap_weighted_sectors_match_voo_and_their_history(self):
        behaviors = pd.read_excel(WORKBOOK, sheet_name="Asset State Behaviors")
        behaviors = behaviors.set_index(behaviors["Ticker"].map(lambda ticker: str(ticker).strip()))
        weights = np.array([SECTOR_WEIGHTS[ticker] for ticker in SECTORS])
        self.assertAlmostEqual(float(weights.sum()), 1.0)

        assumptions = load_simulation_assumptions(WORKBOOK)
        factor = load_asset_cholesky(sheet_name=SHEET)
        labels = list(factor.index)
        baseline = factor.to_numpy() @ factor.to_numpy().T
        matrices = regime_correlation_matrices(baseline, labels, assumptions)
        positions = [labels.index(ticker) for ticker in SECTORS]
        voo = behaviors.loc["VOO"]
        states = (
            ("State 1 Drift", "State 1 Volatility", matrices[0]),
            ("State 2 Drift", "State 2 Volatility", matrices[1]),
            ("State 3 Drift", "State 3 Volatility", matrices[2]),
        )

        drifts = []
        vols = []
        for drift_name, vol_name, matrix in states:
            drift = behaviors.loc[SECTORS, drift_name].to_numpy(dtype=float)
            vol = behaviors.loc[SECTORS, vol_name].to_numpy(dtype=float)
            correlation = matrix[np.ix_(positions, positions)]
            variance = weights @ ((vol[:, None] * correlation * vol[None, :]) @ weights)
            np.testing.assert_allclose(weights @ drift, float(voo[drift_name]), atol=5e-4)
            np.testing.assert_allclose(np.sqrt(variance), float(voo[vol_name]), atol=5e-4)
            drifts.append(drift)
            vols.append(vol)

        expansion, churn, liquidation = drifts
        expansion_vol, churn_vol, liquidation_vol = vols
        order = {ticker: index for index, ticker in enumerate(SECTORS)}

        # Calm up-markets were led by technology and lagged by staples.
        self.assertEqual(int(np.argmax(expansion)), order["XLK"])
        self.assertEqual(int(np.argmin(expansion)), order["XLP"])
        self.assertGreater(expansion[order["XLK"]], float(voo["State 1 Drift"]))
        self.assertLess(expansion[order["XLP"]], float(voo["State 1 Drift"]))
        self.assertEqual(int(np.argmax(expansion_vol)), order["XLE"])
        self.assertEqual(int(np.argmin(expansion_vol)), order["XLP"])

        # Choppy markets kept energy as the volatile sleeve, staples as the calm one.
        self.assertEqual(int(np.argmax(churn_vol)), order["XLE"])
        self.assertEqual(int(np.argmin(churn_vol)), order["XLP"])
        self.assertGreater(churn[order["XLE"]], float(voo["State 2 Drift"]))

        # Down markets took every sector with them. Defensives fell less, cyclicals more.
        self.assertTrue(np.all(liquidation < 0.0))
        self.assertGreater(liquidation[order["XLU"]], float(voo["State 3 Drift"]))
        self.assertGreater(liquidation[order["XLV"]], float(voo["State 3 Drift"]))
        self.assertGreater(liquidation[order["XLP"]], float(voo["State 3 Drift"]))
        self.assertLess(liquidation[order["XLY"]], float(voo["State 3 Drift"]))
        self.assertLess(liquidation[order["XLF"]], float(voo["State 3 Drift"]))
        self.assertLess(liquidation[order["XLI"]], float(voo["State 3 Drift"]))
        self.assertEqual(int(np.argmax(liquidation_vol)), order["XLE"])
        self.assertTrue(np.all(expansion_vol < churn_vol))
        self.assertTrue(np.all(churn_vol < liquidation_vol))
