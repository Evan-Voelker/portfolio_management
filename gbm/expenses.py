import numpy as np
import pandas as pd

EXPENSE_ALIASES = {
    "BTC": "IBIT",
    "IBIT": "BTC",
    "VIIIX": "VOO",
}


def load_expense_ratios(path, sheet="Net Expense Ratios"):
    table = pd.read_excel(path, sheet_name=sheet)
    return dict(zip(table["Ticker"], table["Net Expense Ratio"]))


def overlay_model_expenses(ratios, path, sheet_name):
    """A literal Net Expense Ratio on the model sheet replaces the fee table.

    A formula is left to the fee table. That keeps a lookup from freezing
    whatever value Excel last cached. Tickers with no numeric cell keep the
    table, and a missing column leaves the table unchanged.
    """
    import openpyxl

    ratios = {str(ticker): float(ratio) for ticker, ratio in ratios.items()}
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        if sheet_name not in workbook.sheetnames:
            return ratios
        rows = workbook[sheet_name].iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            return ratios
        names = [None if cell is None else str(cell).strip() for cell in header]
        try:
            ticker_column = names.index("Ticker")
            expense_column = names.index("Net Expense Ratio")
        except ValueError:
            return ratios
        for row in rows:
            if row is None or ticker_column >= len(row) or expense_column >= len(row):
                continue
            ticker = row[ticker_column]
            value = row[expense_column]
            if ticker is None or isinstance(value, str):
                continue
            number = _finite_ratio(value)
            if number is None:
                continue
            ratios[str(ticker).strip()] = number
    finally:
        workbook.close()
    return ratios


def _finite_ratio(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def apply_expense_ratios(drift, labels, ratios):
    """Annual drift net of each asset's expense ratio."""
    ratios = {str(ticker): float(ratio) for ticker, ratio in ratios.items()}
    expense = np.array([_expense_for_label(label, ratios) for label in labels], dtype=float)
    return np.asarray(drift, dtype=float) - expense


def _expense_for_label(label, ratios):
    candidates = [str(label).strip()]
    candidates.extend(part.strip() for part in str(label).split("/"))
    candidates.extend(EXPENSE_ALIASES.get(candidate, "") for candidate in list(candidates))
    for candidate in candidates:
        if candidate in ratios:
            return ratios[candidate]
    return 0.0
