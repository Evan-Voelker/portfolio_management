from pathlib import Path

import numpy as np
import pandas as pd

from gbm.tqqq import TQQQ_EXPENSE, TQQQ_LEVERAGE

# Sheet order. VOO covers the VOO/VIIX sleeve. BTC is the ETF backfilled with spot.
# SGOV is the ETF backfilled with the 3-month bill before it listed.
# VXUS is the ETF share class, backfilled with the 1996 investor class of the same fund.
# VOO is the ETF share class, backfilled with VFINX. SPY starts later, so it is not used.
# CSVYX is the 2009 institutional class, backfilled with the same fund's older classes.
CORRELATION_TICKERS = ["VOO", "SPMO", "DWAS", "CSVYX", "AVES", "TQQQ", "BTC", "DBMF", "KMLM", "GLD", "SGOV"]
# CBOE 13-week Treasury bill. The quote is an annualized yield in percent.
THREE_MONTH_YIELD = "^IRX"
# Investor shares of Vanguard Total International Stock Index Fund, the same
# fund as VXUS. The fund's benchmark is the spliced Total International Stock
# Index: the composite through August 2006, MSCI EAFE + Emerging Markets through
# December 2010, MSCI ACWI ex USA IMI through June 2013, and FTSE Global All Cap
# ex US after that.
INTERNATIONAL_FUND = "VGTSX"
# Investor shares of Vanguard 500 Index Fund, the same fund as VOO. The
# downloaded history starts in 1980. SPY's first print is 1993, so the
# investor class is the longer S&P 500 series.
SP500_FUND = "VFINX"
# Columbia Small Cap Value Discovery Fund. CSVYX is Institutional 3, from
# July 2009. CSCZX is the institutional class from July 1995, and CSMIX is
# class A from July 1986. Both hold the same portfolio.
SMALL_VALUE_INSTITUTIONAL = "CSCZX"
SMALL_VALUE_CLASS_A = "CSMIX"
# Nasdaq-100. QQQ is the total-return ETF from 1999. ^NDX is the price index
# and covers 1996, which is before QQQ listed. TQQQ itself listed in 2010.
NASDAQ_100 = "^NDX"
NASDAQ_100_FUND = "QQQ"
# Bond-equivalent yields accrue on an actual/365 calendar.
BILL_YEAR_DAYS = 365
CHOLESKY_PATH = Path(__file__).resolve().with_name("asset_cholesky.csv")
WORKBOOK_PATH = Path(__file__).resolve().parents[1] / "portfolio_gbm_calculator.xlsx"


def cholesky_from_correlation(correlation):
    """Lower-triangular factor of a positive-definite repair of `correlation`."""
    labels = list(correlation.index)
    values = np.array(correlation.to_numpy(dtype=float), copy=True)
    np.fill_diagonal(values, 1.0)
    values[np.isnan(values)] = 0.0
    values = (values + values.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(values)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    factor = np.linalg.cholesky(repaired)
    return pd.DataFrame(factor, index=labels, columns=list(correlation.columns))


def closes_from_download(raw, tickers):
    """Close prices from a yfinance download, columns in `tickers` order."""
    tickers = list(tickers)
    if isinstance(raw, pd.Series):
        return raw.to_frame(name=tickers[0])

    if isinstance(raw.columns, pd.MultiIndex):
        levels = [set(raw.columns.get_level_values(level)) for level in range(raw.columns.nlevels)]
        if "Close" in levels[0]:
            close = raw["Close"]
        elif "Close" in levels[-1]:
            close = raw.xs("Close", axis=1, level=-1)
        else:
            raise ValueError("price download has no Close column")
    elif "Close" in raw.columns:
        close = raw[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = tickers
    else:
        close = raw

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.reindex(columns=tickers)


def combine_bitcoin(etf, spot):
    """Spot history scaled onto the first ETF print, then the ETF itself.

    Spot and the ETF share trade at different levels. The scale is the ETF
    close divided by the spot close on the first day both exist, applied to
    every earlier spot print. The join day keeps that day's spot return and
    the ETF's price level.
    """
    etf = etf.copy()
    spot = spot.copy()
    both = etf.notna() & spot.notna()
    if both.any():
        first = both[both].index[0]
        spot_close = float(spot.loc[first])
        etf_close = float(etf.loc[first])
        if np.isfinite(spot_close) and np.isfinite(etf_close) and spot_close != 0:
            spot = spot * (etf_close / spot_close)
    combined = etf.combine_first(spot)
    combined.name = "BTC"
    return combined


def three_month_bill_index(yields):
    """Total-return level of a rolled 3-month bill.

    `yields` is an annualized percent yield, the CBOE 13-week bill (^IRX),
    not a price. Each business day accrues the previous yield across the
    calendar days since the previous session, so a weekend earns three days.
    A missing print is filled from the last yield. The level starts at 1.
    """
    yields = _naive_price_series(pd.Series(yields, dtype=float))
    yields = yields[np.isfinite(yields.to_numpy(dtype=float))]
    if yields.empty:
        raise ValueError("3-month bill yield is empty")
    calendar = pd.bdate_range(yields.index.min(), yields.index.max())
    yields = yields.reindex(calendar).ffill()
    gap = yields.index.to_series().diff().dt.days.fillna(0).to_numpy(dtype=float)
    rate = yields.shift(1).to_numpy(dtype=float) / 100.0
    growth = np.ones(len(yields), dtype=float)
    usable = np.isfinite(rate) & (gap > 0)
    growth[usable] = 1.0 + rate[usable] * (gap[usable] / BILL_YEAR_DAYS)
    if np.any(growth <= 0):
        raise ValueError("3-month bill yield produces a non-positive level")
    return pd.Series(np.cumprod(growth), index=yields.index, name="SGOV")


def combine_sgov(etf, bills):
    """Bill-index history scaled onto the first SGOV print, then SGOV itself.

    The bill series is a total-return level and SGOV trades near 100. The
    scale is the SGOV close divided by the bill level on the first day both
    exist, applied to every earlier bill print. The join day keeps that day's
    bill return and SGOV's price level. Later days are SGOV.
    """
    etf = _naive_price_series(etf)
    bills = _naive_price_series(bills)
    both = etf.notna() & bills.notna()
    if both.any():
        first = both[both].index[0]
        bill_level = float(bills.loc[first])
        etf_close = float(etf.loc[first])
        if np.isfinite(bill_level) and np.isfinite(etf_close) and bill_level != 0:
            bills = bills * (etf_close / bill_level)
    combined = etf.combine_first(bills)
    combined.name = "SGOV"
    return combined


def combine_vxus(etf, fund):
    """Investor-share history scaled onto the first VXUS print, then VXUS itself.

    VGTSX is the 1996 investor class of the same total international fund, so
    its closes follow the spliced index the fund tracked before the ETF listed.
    The scale is the VXUS close divided by the fund close on the first day both
    exist, applied to every earlier fund print. The join day keeps that day's
    fund return and VXUS's price level. Later days are VXUS.
    """
    etf = _naive_price_series(etf)
    fund = _naive_price_series(fund)
    both = etf.notna() & fund.notna()
    if both.any():
        first = both[both].index[0]
        fund_close = float(fund.loc[first])
        etf_close = float(etf.loc[first])
        if np.isfinite(fund_close) and np.isfinite(etf_close) and fund_close != 0:
            fund = fund * (etf_close / fund_close)
    combined = etf.combine_first(fund)
    combined.name = "VXUS"
    return combined


def combine_voo(etf, fund):
    """Investor-share history scaled onto the first VOO print, then VOO itself.

    VFINX is the investor class of the same 500 Index Fund. Its history starts
    before SPY, so the backfill uses VFINX. The scale is the VOO close divided
    by the fund close on the first day both exist, applied to every earlier
    fund print. The join day keeps that day's fund return and VOO's price
    level. Later days are VOO.
    """
    etf = _naive_price_series(etf)
    fund = _naive_price_series(fund)
    both = etf.notna() & fund.notna()
    if both.any():
        first = both[both].index[0]
        fund_close = float(fund.loc[first])
        etf_close = float(etf.loc[first])
        if np.isfinite(fund_close) and np.isfinite(etf_close) and fund_close != 0:
            fund = fund * (etf_close / fund_close)
    combined = etf.combine_first(fund)
    combined.name = "VOO"
    return combined


def leveraged_reset_level(closes, financing_percent, leverage=TQQQ_LEVERAGE, expense=TQQQ_EXPENSE):
    """Price level of a daily-reset leveraged fund, starting at 1.

    `closes` is the underlying total-return level. `financing_percent` is an
    annualized percent yield, the same quote as the 3-month bill. Each session
    earns `leverage` times the underlying simple return, then pays the expense
    ratio and the financing rate on the borrowed `(leverage - 1)` dollars.
    Both costs accrue over the calendar gap since the previous session, so a
    weekend is three days. A session that would wipe the fund out is floored
    just above zero. The first print is 1 and has no return.
    """
    closes = _naive_price_series(closes).dropna()
    if closes.empty:
        raise ValueError("leveraged index is empty")
    yields = _naive_price_series(financing_percent).reindex(closes.index).ffill().bfill()
    gap = closes.index.to_series().diff().dt.days.to_numpy(dtype=float)
    simple = closes.pct_change().to_numpy(dtype=float)
    rate = yields.shift(1).to_numpy(dtype=float) / 100.0
    year_fraction = np.zeros(len(closes), dtype=float)
    usable = np.isfinite(gap) & (gap > 0)
    year_fraction[usable] = gap[usable] / BILL_YEAR_DAYS
    rate = np.where(np.isfinite(rate), rate, 0.0)
    simple = np.where(np.isfinite(simple), simple, 0.0)
    leveraged = (
        float(leverage) * simple
        - (float(leverage) - 1.0) * rate * year_fraction
        - float(expense) * year_fraction
    )
    growth = np.maximum(1.0 + leveraged, 1e-6)
    growth[0] = 1.0
    return pd.Series(np.cumprod(growth), index=closes.index, name="TQQQ")


def combine_tqqq(etf, synthetic):
    """Synthetic 3x Nasdaq history scaled onto the first TQQQ print, then TQQQ.

    TQQQ listed in 2010. The synthetic series is the daily-reset 3x path of
    the Nasdaq-100 back to 1996. The scale is the TQQQ close divided by the
    synthetic level on the first day both exist, applied to every earlier
    synthetic print. The join day keeps that day's synthetic return and
    TQQQ's price level. Later days are TQQQ.
    """
    combined = _scale_prior(etf, synthetic)
    combined.name = "TQQQ"
    return combined


def combine_csvyx(share_class, institutional, class_a=None):
    """Older classes of the same small-cap value fund, then CSVYX itself.

    CSCZX starts in July 1995 and CSMIX starts in July 1986, so the series
    reaches 1996. Each older class is scaled onto the first print of the next
    one. The join day keeps the older class's return and the newer class's
    price. From CSVYX's listing, the series is CSVYX.
    """
    history = _scale_prior(share_class, institutional)
    if class_a is not None:
        history = _scale_prior(history, class_a)
    history.name = "CSVYX"
    return history


def _scale_prior(current, prior):
    """Scale `prior` onto the first shared print, then keep `current` after it."""
    current = _naive_price_series(current)
    prior = _naive_price_series(prior)
    both = current.notna() & prior.notna()
    if both.any():
        first = both[both].index[0]
        prior_close = float(prior.loc[first])
        current_close = float(current.loc[first])
        if np.isfinite(prior_close) and np.isfinite(current_close) and prior_close != 0:
            prior = prior * (current_close / prior_close)
    return current.combine_first(prior)


def _naive_price_series(series):
    """Daily series on a naive midnight index. A repeated session keeps the last print."""
    series = pd.Series(series, dtype=float).copy()
    index = pd.DatetimeIndex(pd.to_datetime(series.index))
    if index.tz is not None:
        index = index.normalize().tz_localize(None)
    else:
        index = index.normalize()
    series.index = index
    series = series.groupby(level=0).last()
    series.index.name = None
    return series.sort_index()


def download_prices(tickers, start=None):
    """Daily adjusted closes.

    The default request is the last 20 years, which is what the correlation
    factor is built from. `start` asks for history from that date forward so
    a backtest can reach past the 20-year window. Bitcoin is the ETF backfilled
    with spot. SGOV is the ETF backfilled with the 3-month bill index. VXUS is
    the ETF backfilled with VGTSX, the original investor class of the same fund.
    VOO is the ETF backfilled with VFINX, which starts before SPY. CSVYX is
    backfilled with CSCZX and CSMIX, the older classes of the same fund.
    TQQQ is the ETF backfilled with a daily-reset 3x Nasdaq-100. QQQ supplies
    the total return from 1999, and the Nasdaq-100 price index covers 1996.
    """
    import yfinance as yf

    tickers = list(tickers)
    request = {"auto_adjust": True, "progress": False}
    if start is None:
        request["period"] = "20y"
    else:
        request["start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
    frames = []
    equity = [ticker for ticker in tickers if ticker not in ("BTC", "SGOV", "VXUS", "VOO", "CSVYX", "TQQQ")]
    if equity:
        frames.append(closes_from_download(yf.download(equity, **request), equity))
    if "VOO" in tickers:
        etf = closes_from_download(yf.download("VOO", **request), ["VOO"]).iloc[:, 0]
        fund = closes_from_download(yf.download(SP500_FUND, **request), [SP500_FUND]).iloc[:, 0]
        frames.append(combine_voo(etf, fund).to_frame())
    if "CSVYX" in tickers:
        share_class = closes_from_download(yf.download("CSVYX", **request), ["CSVYX"]).iloc[:, 0]
        institutional = closes_from_download(
            yf.download(SMALL_VALUE_INSTITUTIONAL, **request), [SMALL_VALUE_INSTITUTIONAL]
        ).iloc[:, 0]
        class_a = closes_from_download(
            yf.download(SMALL_VALUE_CLASS_A, **request), [SMALL_VALUE_CLASS_A]
        ).iloc[:, 0]
        frames.append(combine_csvyx(share_class, institutional, class_a).to_frame())
    if "VXUS" in tickers:
        etf = closes_from_download(yf.download("VXUS", **request), ["VXUS"]).iloc[:, 0]
        fund = closes_from_download(yf.download(INTERNATIONAL_FUND, **request), [INTERNATIONAL_FUND]).iloc[:, 0]
        frames.append(combine_vxus(etf, fund).to_frame())
    if "BTC" in tickers:
        etf = closes_from_download(yf.download("BTC", **request), ["BTC"]).iloc[:, 0]
        spot = closes_from_download(yf.download("BTC-USD", **request), ["BTC-USD"]).iloc[:, 0]
        frames.append(combine_bitcoin(etf, spot).to_frame())
    if "TQQQ" in tickers:
        etf = closes_from_download(yf.download("TQQQ", **request), ["TQQQ"]).iloc[:, 0]
        qqq = closes_from_download(yf.download(NASDAQ_100_FUND, **request), [NASDAQ_100_FUND]).iloc[:, 0]
        index = closes_from_download(yf.download(NASDAQ_100, **request), [NASDAQ_100]).iloc[:, 0]
        bill_yield = closes_from_download(yf.download(THREE_MONTH_YIELD, **request), [THREE_MONTH_YIELD]).iloc[:, 0]
        nasdaq = _scale_prior(qqq, index)
        frames.append(combine_tqqq(etf, leveraged_reset_level(nasdaq, bill_yield)).to_frame())
    if "SGOV" in tickers:
        etf = closes_from_download(yf.download("SGOV", **request), ["SGOV"]).iloc[:, 0]
        bill_yield = closes_from_download(yf.download(THREE_MONTH_YIELD, **request), [THREE_MONTH_YIELD]).iloc[:, 0]
        frames.append(combine_sgov(etf, three_month_bill_index(bill_yield)).to_frame())
    return pd.concat(frames, axis=1).reindex(columns=tickers)


def market_ticker(label):
    """Correlation symbol for a sleeve. VOO/VIIX is VOO, and BTC keeps the spot backfill."""
    text = str(label).strip()
    if text in ("BTC", "IBIT"):
        return "BTC"
    parts = [part.strip() for part in text.split("/") if part.strip()]
    return parts[0] if parts else text


def tickers_from_sheet(sheet_name, workbook=None):
    """Unique market tickers in sheet order."""
    workbook = WORKBOOK_PATH if workbook is None else Path(workbook)
    frame = pd.read_excel(workbook, sheet_name=sheet_name)
    if "Ticker" not in frame.columns:
        raise ValueError(f"{sheet_name} has no Ticker column")
    tickers = []
    for label in frame["Ticker"].tolist():
        if label is None or (isinstance(label, float) and np.isnan(label)):
            continue
        key = market_ticker(label)
        if key and key not in tickers:
            tickers.append(key)
    if not tickers:
        raise ValueError(f"{sheet_name} has no tickers")
    return tickers


def cholesky_path_for(sheet_name=None):
    """Default factor file, or one file per model sheet so they do not overwrite each other."""
    if not sheet_name:
        return CHOLESKY_PATH
    slug = []
    for character in str(sheet_name).strip().lower():
        slug.append(character if character.isalnum() else "_")
    name = "".join(slug).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return CHOLESKY_PATH.with_name(f"asset_cholesky_{name}.csv")


def generate_correlation_and_cholesky(tickers=None, sheet_name=None, workbook=None):
    """Pairwise return correlation for `tickers`, repaired and factored.

    `sheet_name` reads the tickers from that Excel sheet and ignores `tickers`.
    """
    if sheet_name:
        tickers = tickers_from_sheet(sheet_name, workbook)
    else:
        tickers = list(CORRELATION_TICKERS if tickers is None else tickers)
    print(f"Downloading 20 years of prices for {', '.join(tickers)}...")
    prices = download_prices(tickers)
    correlation = prices.pct_change().corr(method="pearson")
    correlation = correlation.reindex(index=tickers, columns=tickers)
    return cholesky_from_correlation(correlation)


def load_asset_cholesky(tickers=None, path=None, generate=None, sheet_name=None, workbook=None):
    """Cholesky factor for `tickers`. Rebuild only when the file is missing or the wrong size.

    `sheet_name` selects the model. Its tickers come from that sheet, and the factor
    is stored beside the default file under a name derived from the sheet.
    """
    if sheet_name:
        tickers = tickers_from_sheet(sheet_name, workbook)
    else:
        tickers = list(CORRELATION_TICKERS if tickers is None else tickers)
    path = cholesky_path_for(sheet_name) if path is None else Path(path)
    saved = _read_cholesky(path)
    if saved is not None and _factor_matches(saved, tickers):
        return saved

    if generate is None:
        generate = generate_correlation_and_cholesky
    factor = _as_factor_frame(generate(tickers), tickers)
    if not _factor_matches(factor, tickers):
        raise ValueError(
            f"Cholesky factor has shape {factor.shape} for {list(factor.index)}, expected {len(tickers)} {tickers}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    factor.to_csv(path)
    return factor


def _factor_matches(frame, tickers):
    return (
        frame.shape == (len(tickers), len(tickers))
        and list(frame.index) == list(tickers)
        and list(frame.columns) == list(tickers)
    )


def _read_cholesky(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, index_col=0)
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    return _as_factor_frame(frame, list(frame.index))


def _as_factor_frame(factor, tickers):
    if not isinstance(factor, pd.DataFrame):
        factor = pd.DataFrame(np.asarray(factor, dtype=float), index=tickers, columns=tickers)
    else:
        factor = factor.copy()
    factor.index = factor.index.map(lambda value: str(value).strip())
    factor.columns = factor.columns.map(lambda value: str(value).strip())
    factor.index.name = None
    return factor.astype(float)


def save_cholesky_plots(factor, output_path="correlation_cholesky_plots.png"):
    """Heatmaps of the recovered correlation and its Cholesky factor."""
    import matplotlib.pyplot as plt

    correlation = factor.to_numpy() @ factor.to_numpy().T
    lower = np.array(factor.to_numpy(), copy=True)
    lower[np.triu(np.ones(lower.shape, dtype=bool), k=1)] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    image = axes[0].imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Recovered correlation (L @ L.T)")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    factor_image = axes[1].imshow(lower, cmap="Blues")
    axes[1].set_title("Cholesky factor (L)")
    fig.colorbar(factor_image, ax=axes[1], fraction=0.046)
    for axis, values in ((axes[0], correlation), (axes[1], lower)):
        axis.set_xticks(range(len(factor.columns)), list(factor.columns), rotation=45, ha="right")
        axis.set_yticks(range(len(factor.index)), list(factor.index))
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                if np.isnan(values[row, column]):
                    continue
                axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the correlation Cholesky factor for one model sheet.")
    parser.add_argument("--sheet-name", default=None, help="Excel sheet whose tickers define the matrix")
    parser.add_argument("--workbook", default=None)
    args = parser.parse_args()
    save_cholesky_plots(load_asset_cholesky(sheet_name=args.sheet_name, workbook=args.workbook))
