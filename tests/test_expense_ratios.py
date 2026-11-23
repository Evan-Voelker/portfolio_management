import unittest

import numpy as np
import openpyxl

from gbm.expenses import apply_expense_ratios

WORKBOOK = r"C:\Users\Evan\VirtualBox VMs\Sandbox\PortfolioManagement\portfolio_gbm_calculator.xlsx"

STARTING_EXPENSES = {
    "VOO": ("Vanguard S&P 500 ETF", 0.0003),
    "IBIT": ("iShares Bitcoin Trust", 0.0025),
    "AVES": ("Avantis Emerging Markets Value ETF", 0.0036),
    "CSVYX": ("Columbia Small Cap Value Discovery Fund", 0.0081),
    "DBMF": ("iMGP DBi Managed Futures Strategy ETF", 0.0085),
    "TQQQ": ("ProShares UltraPro QQQ (3x Leverage)", 0.0086),
    "KMLM": ("KraneShares Mount Lucas Managed Futures ETF", 0.0090),
    "GLD": ("SPDR Gold Shares", 0.0040),
    "SGOV": ("US Funds", 0.0030),
}


class ExpenseRatioTableTest(unittest.TestCase):
    def test_workbook_table_stores_the_starting_expense_ratios(self):
        workbook = openpyxl.load_workbook(WORKBOOK)
        sheet = workbook["Net Expense Ratios"]
        table = sheet.tables["ExpenseRatios"]

        self.assertEqual(table.ref, "A1:C10")
        self.assertEqual([cell.value for cell in sheet[1]], ["Ticker", "Fund / Asset Name", "Net Expense Ratio"])
        for row in range(2, 11):
            ticker = sheet.cell(row, 1).value
            name, ratio = STARTING_EXPENSES[ticker]
            self.assertEqual(sheet.cell(row, 2).value, name)
            self.assertAlmostEqual(sheet.cell(row, 3).value, ratio)
            self.assertEqual(sheet.cell(row, 3).number_format, "0.00%")

    def test_asset_parameters_look_up_each_expense_ratio(self):
        workbook = openpyxl.load_workbook(WORKBOOK)
        sheet = workbook["Asset Parameters"]

        self.assertEqual(sheet["I1"].value, "Tracked Index")
        self.assertEqual(sheet["J1"].value, "Net Expense Ratio")
        self.assertIn("ExpenseRatios[Ticker]", sheet["J2"].value)
        self.assertEqual(sheet["J2"].number_format, "0.00%")

    def test_price_path_drift_is_net_of_the_expense_ratio(self):
        workbook = openpyxl.load_workbook(WORKBOOK)
        formula = workbook["GBM Price Path Generator"]["C4"].value

        self.assertIn("'Asset Parameters'!$J2", formula)

    def test_btc_and_voo_aliases_reduce_annual_drift(self):
        ratios = {ticker: ratio for ticker, (_name, ratio) in STARTING_EXPENSES.items()}
        net = apply_expense_ratios([0.10, 0.25, 0.11], ["VOO/VIIX", "BTC", "CSVYX"], ratios)

        np.testing.assert_allclose(net, [0.10 - 0.0003, 0.25 - 0.0025, 0.11 - 0.0081])


if __name__ == "__main__":
    unittest.main()
