import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gbm.asset_simulation_matrix import (
    CHOLESKY_PATH,
    cholesky_from_correlation,
    cholesky_path_for,
    closes_from_download,
    combine_bitcoin,
    load_asset_cholesky,
    tickers_from_sheet,
)


class CholeskyFactorTest(unittest.TestCase):
    def test_repairs_an_indefinite_correlation_into_a_unit_diagonal_factor(self):
        labels = ["VOO", "DBMF", "BTC"]
        correlation = pd.DataFrame(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            index=labels,
            columns=labels,
        )

        factor = cholesky_from_correlation(correlation)

        self.assertEqual(list(factor.index), labels)
        self.assertEqual(list(factor.columns), labels)
        np.testing.assert_allclose(np.triu(factor.to_numpy(), k=1), np.zeros((3, 3)), atol=1e-8)
        recovered = factor.to_numpy() @ factor.to_numpy().T
        np.testing.assert_allclose(np.diag(recovered), np.ones(3), atol=1e-8)
        self.assertTrue(np.all(np.linalg.eigvalsh(recovered) > 0))

    def test_missing_pairwise_entries_stay_uncorrelated(self):
        labels = ["VOO", "SGOV"]
        correlation = pd.DataFrame(
            [[1.0, np.nan], [np.nan, 1.0]],
            index=labels,
            columns=labels,
        )

        factor = cholesky_from_correlation(correlation)

        recovered = factor.to_numpy() @ factor.to_numpy().T
        np.testing.assert_allclose(recovered, np.eye(2), atol=1e-8)


class SavedCholeskyTest(unittest.TestCase):
    def test_loads_a_saved_factor_of_the_right_size_without_regenerating(self):
        tickers = ["VOO", "BTC"]
        saved = pd.DataFrame([[1.0, 0.0], [0.4, 0.9165]], index=tickers, columns=tickers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset_cholesky.csv"
            saved.to_csv(path)
            calls = []

            loaded = load_asset_cholesky(tickers, path, generate=lambda names: calls.append(list(names)))

        self.assertEqual(calls, [])
        pd.testing.assert_frame_equal(loaded, saved)

    def test_regenerates_and_saves_when_the_file_is_missing(self):
        tickers = ["VOO", "BTC"]
        built = pd.DataFrame([[1.0, 0.0], [0.2, 0.98]], index=tickers, columns=tickers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset_cholesky.csv"

            loaded = load_asset_cholesky(tickers, path, generate=lambda names: built)

            self.assertTrue(path.exists())
            pd.testing.assert_frame_equal(loaded, built)
            stored = pd.read_csv(path, index_col=0)
            stored.index = stored.index.astype(str)
            stored.columns = stored.columns.astype(str)
            pd.testing.assert_frame_equal(stored, built)

    def test_regenerates_when_the_saved_matrix_is_the_wrong_size(self):
        tickers = ["VOO", "BTC", "DBMF"]
        wrong = pd.DataFrame(np.eye(2), index=["VOO", "BTC"], columns=["VOO", "BTC"])
        built = pd.DataFrame(np.eye(3), index=tickers, columns=tickers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset_cholesky.csv"
            wrong.to_csv(path)
            calls = []

            def generate(names):
                calls.append(list(names))
                return built

            loaded = load_asset_cholesky(tickers, path, generate=generate)
            stored = pd.read_csv(path, index_col=0)

        self.assertEqual(calls, [tickers])
        pd.testing.assert_frame_equal(loaded, built)
        self.assertEqual(stored.shape, (3, 3))

    def test_regenerates_when_the_labels_do_not_match_the_tickers(self):
        tickers = ["VOO", "BTC"]
        wrong = pd.DataFrame(np.eye(2), index=["GLD", "SGOV"], columns=["GLD", "SGOV"])
        built = pd.DataFrame([[1.0, 0.0], [0.3, 0.95]], index=tickers, columns=tickers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset_cholesky.csv"
            wrong.to_csv(path)

            loaded = load_asset_cholesky(tickers, path, generate=lambda names: built)

        pd.testing.assert_frame_equal(loaded, built)


class SheetCholeskyTest(unittest.TestCase):
    def test_sheet_name_selects_tickers_and_a_separate_factor_file(self):
        tickers = ["XLK", "BTC", "VOO"]
        built = pd.DataFrame(np.eye(3), index=tickers, columns=tickers)
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "book.xlsx"
            pd.DataFrame({"Ticker": ["XLK", "BTC", "VOO/VIIX"]}).to_excel(
                workbook, sheet_name="Sector Momentum Playbook", index=False
            )
            path = Path(directory) / "factor.csv"
            seen = {}

            def generate(names):
                seen["names"] = list(names)
                return built

            loaded = load_asset_cholesky(
                path=path,
                generate=generate,
                sheet_name="Sector Momentum Playbook",
                workbook=workbook,
            )
            self.assertEqual(tickers_from_sheet("Sector Momentum Playbook", workbook), tickers)

        self.assertEqual(seen["names"], tickers)
        pd.testing.assert_frame_equal(loaded, built)
        self.assertNotEqual(cholesky_path_for("Sector Momentum Playbook"), CHOLESKY_PATH)
        self.assertEqual(
            cholesky_path_for("Sector Momentum Playbook").name,
            "asset_cholesky_sector_momentum_playbook.csv",
        )


class DownloadShapeTest(unittest.TestCase):
    def test_close_frame_reads_a_multiindex_download(self):
        raw = pd.DataFrame(
            [[10.0, 20.0, 1.0, 2.0]],
            columns=pd.MultiIndex.from_product([["Close", "Open"], ["VOO", "GLD"]]),
        )

        closes = closes_from_download(raw, ["VOO", "GLD"])

        self.assertEqual(list(closes.columns), ["VOO", "GLD"])
        np.testing.assert_allclose(closes.iloc[0].to_numpy(), [10.0, 20.0])

    def test_bitcoin_history_prefers_the_etf_and_backfills_spot(self):
        index = pd.to_datetime(["2024-01-01", "2024-01-02"])
        etf = pd.Series([np.nan, 20.0], index=index)
        spot = pd.Series([10.0, 99.0], index=index)

        combined = combine_bitcoin(etf, spot)

        np.testing.assert_allclose(combined.to_numpy(), [10.0 * 20.0 / 99.0, 20.0])
        self.assertEqual(combined.name, "BTC")


class TreasuryBillBackfillTest(unittest.TestCase):
    def test_a_yield_accrues_across_the_calendar_gap(self):
        from gbm.asset_simulation_matrix import three_month_bill_index

        # Friday to Monday is three calendar days, not one session.
        yields = pd.Series([3.65, 3.65], index=pd.to_datetime(["2020-01-03", "2020-01-06"]))

        level = three_month_bill_index(yields)

        self.assertEqual(level.index[0], pd.Timestamp("2020-01-03"))
        self.assertEqual(level.index[-1], pd.Timestamp("2020-01-06"))
        np.testing.assert_allclose(level.iloc[0], 1.0)
        np.testing.assert_allclose(level.iloc[-1], 1.0 + 0.0365 * 3 / 365)

    def test_a_missing_print_still_accrues_each_business_day(self):
        from gbm.asset_simulation_matrix import three_month_bill_index

        yields = pd.Series(
            [3.65, np.nan, 3.65],
            index=pd.to_datetime(["2020-01-06", "2020-01-08", "2020-01-10"]),
        )

        level = three_month_bill_index(yields)

        self.assertEqual(len(level), 5)
        np.testing.assert_allclose(level.iloc[0], 1.0)
        np.testing.assert_allclose(level.iloc[-1], (1.0 + 0.0365 / 365) ** 4)

    def test_a_zero_yield_does_not_move_the_level(self):
        from gbm.asset_simulation_matrix import three_month_bill_index

        yields = pd.Series([0.0, 0.0], index=pd.to_datetime(["2020-01-03", "2020-01-06"]))

        level = three_month_bill_index(yields)

        np.testing.assert_allclose(level.to_numpy(), [1.0, 1.0])

    def test_bill_history_is_scaled_onto_the_first_sgov_print(self):
        from gbm.asset_simulation_matrix import combine_sgov

        index = pd.to_datetime(["2020-05-22", "2020-05-26", "2020-05-27"])
        etf = pd.Series([np.nan, 100.0, 100.1], index=index)
        bills = pd.Series([1.0, 1.01, 1.02], index=index)

        combined = combine_sgov(etf, bills)

        np.testing.assert_allclose(combined.iloc[0], 1.0 * 100.0 / 1.01)
        np.testing.assert_allclose(combined.iloc[1], 100.0)
        np.testing.assert_allclose(combined.iloc[2], 100.1)
        np.testing.assert_allclose(combined.iloc[1] / combined.iloc[0] - 1.0, 1.01 / 1.0 - 1.0)
        self.assertEqual(combined.name, "SGOV")


class InternationalFundBackfillTest(unittest.TestCase):
    def test_investor_shares_are_scaled_onto_the_first_vxus_print(self):
        from gbm.asset_simulation_matrix import combine_vxus

        index = pd.to_datetime(["2011-01-25", "2011-01-26", "2011-01-27"])
        etf = pd.Series([np.nan, 48.0, 48.2], index=index)
        fund = pd.Series([14.0, 14.2, 14.5], index=index)

        combined = combine_vxus(etf, fund)

        np.testing.assert_allclose(combined.iloc[0], 14.0 * 48.0 / 14.2)
        np.testing.assert_allclose(combined.iloc[1], 48.0)
        np.testing.assert_allclose(combined.iloc[2], 48.2)
        np.testing.assert_allclose(combined.iloc[1] / combined.iloc[0] - 1.0, 14.2 / 14.0 - 1.0)
        self.assertEqual(combined.name, "VXUS")


class Sp500FundBackfillTest(unittest.TestCase):
    def test_investor_shares_are_scaled_onto_the_first_voo_print(self):
        from gbm.asset_simulation_matrix import combine_voo

        index = pd.to_datetime(["2010-09-08", "2010-09-09", "2010-09-10"])
        etf = pd.Series([np.nan, 102.0, 102.4], index=index)
        fund = pd.Series([80.0, 81.0, 81.6], index=index)

        combined = combine_voo(etf, fund)

        np.testing.assert_allclose(combined.iloc[0], 80.0 * 102.0 / 81.0)
        np.testing.assert_allclose(combined.iloc[1], 102.0)
        np.testing.assert_allclose(combined.iloc[2], 102.4)
        np.testing.assert_allclose(combined.iloc[1] / combined.iloc[0] - 1.0, 81.0 / 80.0 - 1.0)
        self.assertEqual(combined.name, "VOO")


class SmallValueFundBackfillTest(unittest.TestCase):
    def test_older_share_classes_are_scaled_onto_csvyx(self):
        from gbm.asset_simulation_matrix import combine_csvyx

        index = pd.to_datetime(["1986-07-25", "1995-07-28", "2009-07-15", "2009-07-16"])
        share_class = pd.Series([np.nan, np.nan, 80.0, 88.0], index=index)
        institutional = pd.Series([np.nan, 50.0, 60.0, 70.0], index=index)
        class_a = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)

        combined = combine_csvyx(share_class, institutional, class_a)

        institutional_level = 50.0 * 80.0 / 60.0
        np.testing.assert_allclose(combined.iloc[1], institutional_level)
        np.testing.assert_allclose(combined.iloc[0], 10.0 * institutional_level / 20.0)
        np.testing.assert_allclose(combined.iloc[2], 80.0)
        np.testing.assert_allclose(combined.iloc[3], 88.0)
        np.testing.assert_allclose(combined.iloc[1] / combined.iloc[0] - 1.0, 20.0 / 10.0 - 1.0)
        np.testing.assert_allclose(combined.iloc[2] / combined.iloc[1] - 1.0, 60.0 / 50.0 - 1.0)
        np.testing.assert_allclose(combined.iloc[3] / combined.iloc[2] - 1.0, 88.0 / 80.0 - 1.0)
        self.assertEqual(combined.name, "CSVYX")
