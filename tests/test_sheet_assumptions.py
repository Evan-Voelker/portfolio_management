import unittest
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from gbm.bitcoin import MATURE_BTC_DRIFT, MATURE_BTC_VOLATILITY, mature_return_center
from gbm.expenses import load_expense_ratios, overlay_model_expenses
from gbm.gbm_step import (
    CASH_FLIGHT_LABEL,
    CHURN_CORRELATION_SCALE,
    LIQUIDATION_BLEND,
    LIQUIDATION_CORRELATION,
    horizon_years,
    load_simulation_assumptions,
    regime_correlation_matrices,
)
from gbm.momentum import align_momentum_drift, overlay_sheet_momentum
from gbm.regime import CASH_FLIGHT_TICKERS, TRANSITION_SHEET

WORKBOOK = Path(__file__).resolve().parents[1] / "portfolio_gbm_calculator.xlsx"


class BehaviorSheetTest(unittest.TestCase):
    def test_workbook_bitcoin_row_is_the_mature_vector_the_run_uses(self):
        frame = pd.read_excel(WORKBOOK, sheet_name="Asset State Behaviors")
        bitcoin = frame.loc[frame["Ticker"].astype(str).str.strip() == "BTC"].iloc[0]

        np.testing.assert_allclose(
            [bitcoin["State 1 Drift"], bitcoin["State 2 Drift"], bitcoin["State 3 Drift"]],
            MATURE_BTC_DRIFT,
        )
        np.testing.assert_allclose(
            [bitcoin["State 1 Volatility"], bitcoin["State 2 Volatility"], bitcoin["State 3 Volatility"]],
            MATURE_BTC_VOLATILITY,
        )

    def test_behavior_sheet_momentum_replaces_the_cache(self):
        cache_lambda, cache_center = align_momentum_drift(
            pd.DataFrame({
                "Ticker": ["VOO", "BTC", "DBMF"],
                "Lambda": [-0.33, 0.40, -0.52],
                "Center": [0.14, 1.38, 0.066],
            }),
            ["VOO", "IBIT", "DBMF", "SGOV"],
        )
        behavior = pd.DataFrame([
            ["BTC", 0.0, mature_return_center()],
            ["DBMF", 0.0, 0.0],
        ], columns=["Ticker", "Momentum Lambda", "Momentum Center"])

        lambda_mom, center = overlay_sheet_momentum(cache_lambda, cache_center, ["VOO", "IBIT", "DBMF", "SGOV"], behavior)

        np.testing.assert_allclose(lambda_mom, [-0.33, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(center, [0.14, mature_return_center(), 0.0, 0.0])

    def test_workbook_momentum_columns_hold_the_coefficients_the_run_uses(self):
        frame = pd.read_excel(WORKBOOK, sheet_name="Asset State Behaviors")
        indexed = frame.set_index(frame["Ticker"].astype(str).str.strip())

        self.assertAlmostEqual(float(indexed.loc["BTC", "Momentum Lambda"]), 0.0)
        self.assertAlmostEqual(float(indexed.loc["BTC", "Momentum Center"]), mature_return_center())
        self.assertAlmostEqual(float(indexed.loc["DBMF", "Momentum Lambda"]), 0.0)
        self.assertAlmostEqual(float(indexed.loc["DBMF", "Momentum Center"]), 0.0)
        self.assertAlmostEqual(float(indexed.loc["SGOV", "Momentum Lambda"]), 0.0)
        self.assertAlmostEqual(float(indexed.loc["VOO", "Momentum Lambda"]), -0.33058305410780975)


class HorizonSheetTest(unittest.TestCase):
    def test_the_first_time_to_maturity_is_the_horizon_in_years(self):
        frame = pd.DataFrame({"Time to Maturity": [20, 7, 3]})

        self.assertEqual(horizon_years(frame), 20.0)

    def test_rows_after_the_first_are_ignored(self):
        frame = pd.DataFrame({
            "Ticker": ["VOO", "SGOV"],
            "Time to Maturity": [12.5, 30],
        })

        self.assertEqual(horizon_years(frame), 12.5)

    def test_a_missing_column_keeps_the_ten_year_default(self):
        self.assertEqual(horizon_years(pd.DataFrame({"Ticker": ["VOO"]})), 10.0)

    def test_a_blank_first_cell_keeps_the_ten_year_default(self):
        frame = pd.DataFrame({"Time to Maturity": [None, 15]})

        self.assertEqual(horizon_years(frame), 10.0)

    def test_asset_parameters_runs_for_its_first_maturity(self):
        frame = pd.read_excel(WORKBOOK, sheet_name="Asset Parameters")

        self.assertEqual(float(frame["Time to Maturity"].iloc[0]), 20.0)
        self.assertEqual(horizon_years(frame), 20.0)

    def test_the_sector_sheet_runs_for_its_first_maturity(self):
        frame = pd.read_excel(WORKBOOK, sheet_name="Sector Momentum Playbook")

        self.assertEqual(float(frame["Time to Maturity"].iloc[0]), 10.0)
        self.assertEqual(horizon_years(frame), 10.0)


class ExpenseSheetTest(unittest.TestCase):
    def test_a_literal_model_expense_replaces_the_fee_table(self):
        ratios = overlay_model_expenses(load_expense_ratios(WORKBOOK), WORKBOOK, "Asset Parameters")

        self.assertAlmostEqual(ratios["VXUS"], 0.0003)
        self.assertAlmostEqual(ratios["SGOV"], 0.003)
        self.assertAlmostEqual(ratios["VOO"], 0.0003)
        self.assertAlmostEqual(ratios["IBIT"], 0.0025)

    def test_asset_parameters_literals_match_the_fee_table(self):
        workbook = openpyxl.load_workbook(WORKBOOK, data_only=False)
        sheet = workbook["Asset Parameters"]
        headers = [cell.value for cell in sheet[1]]
        ticker_column = headers.index("Ticker") + 1
        expense_column = headers.index("Net Expense Ratio") + 1
        table = load_expense_ratios(WORKBOOK)
        aliases = {"BTC": "IBIT", "IBIT": "BTC"}

        for row in range(2, sheet.max_row + 1):
            ticker = sheet.cell(row, ticker_column).value
            value = sheet.cell(row, expense_column).value
            if ticker is None or isinstance(value, str):
                continue
            key = str(ticker).strip()
            expected = table.get(key, table.get(aliases.get(key, ""), None))
            self.assertIsNotNone(expected, key)
            self.assertAlmostEqual(float(value), float(expected), msg=key)


class StressSheetTest(unittest.TestCase):
    def test_workbook_stress_block_matches_the_numbers_the_run_was_using(self):
        workbook = openpyxl.load_workbook(WORKBOOK, data_only=False)
        sheet = workbook[TRANSITION_SHEET]
        cells = {}
        for row in range(1, sheet.max_row + 1):
            label = sheet.cell(row, 1).value
            if label is not None:
                cells[str(label).strip()] = sheet.cell(row, 2).value

        self.assertEqual(cells["Churn Correlation Scale"], CHURN_CORRELATION_SCALE)
        self.assertEqual(cells["Liquidation Blend"], LIQUIDATION_BLEND)
        self.assertEqual(cells["Liquidation Correlation"], LIQUIDATION_CORRELATION)
        self.assertEqual(cells["Cash Flight Tickers"], "SPMO, DWAS")
        self.assertEqual(cells["Churn Volatility Grace Months"], 3)
        self.assertEqual(cells["Churn Volatility Monthly Step"], 0.1)
        self.assertEqual(cells["DBMF Liquidation Equity Correlation"], -0.40)
        self.assertEqual(cells["Bitcoin Stress Correlation"], 0.45)
        assumptions = load_simulation_assumptions(WORKBOOK)
        self.assertEqual(assumptions.cash_flight, CASH_FLIGHT_TICKERS)
        self.assertEqual(assumptions.liquidation_correlation, LIQUIDATION_CORRELATION)

    def test_a_sheet_liquidation_correlation_replaces_the_flat_blend(self):
        with _workbook() as path:
            assumptions = load_simulation_assumptions(path)

            self.assertEqual(assumptions.liquidation_correlation, 0.50)
            baseline = np.array([[1.0, 0.20], [0.20, 1.0]])
            _expansion, _churn, liquidation = regime_correlation_matrices(
                baseline, ["VOO", "GLD"], assumptions
            )

        blended = (1.0 - LIQUIDATION_BLEND) * 0.20 + LIQUIDATION_BLEND * 0.50
        np.testing.assert_allclose(liquidation[0, 1], blended)

    def test_a_blank_cash_flight_cell_keeps_the_behavior_row(self):
        with _workbook(cash_flight="") as path:
            assumptions = load_simulation_assumptions(path)

        self.assertEqual(assumptions.cash_flight, ())


class _workbook:
    def __init__(self, cash_flight="SPMO, DWAS, TQQQ", liquidation_correlation=0.50):
        self.cash_flight = cash_flight
        self.liquidation_correlation = liquidation_correlation
        self.path = None

    def __enter__(self):
        import tempfile

        directory = tempfile.TemporaryDirectory()
        self._directory = directory
        self.path = Path(directory.name) / "book.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = TRANSITION_SHEET
        labels = {
            "Churn Correlation Scale": CHURN_CORRELATION_SCALE,
            "Liquidation Blend": LIQUIDATION_BLEND,
            "Liquidation Correlation": self.liquidation_correlation,
            "Bitcoin Stress Correlation": 0.45,
            "Bitcoin Stress Blend": 0.60,
            "DBMF Expansion Equity Correlation": -0.10,
            "DBMF Churn Equity Correlation": -0.075,
            "DBMF Liquidation Equity Correlation": -0.40,
            "Churn Volatility Grace Months": 3,
            "Churn Volatility Monthly Step": 0.1,
            CASH_FLIGHT_LABEL: self.cash_flight,
        }
        for row, (label, value) in enumerate(labels.items(), start=1):
            sheet.cell(row, 1, label)
            sheet.cell(row, 2, value)
        workbook.save(self.path)
        return self.path

    def __exit__(self, *_args):
        self._directory.cleanup()
