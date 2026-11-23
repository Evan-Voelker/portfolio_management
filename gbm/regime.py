import numpy as np
import openpyxl
import pandas as pd
from openpyxl.worksheet.table import Table, TableStyleInfo

# One month of trading days. The transition matrix is drawn on that cadence.
TRADING_DAYS_PER_MONTH = 21
CHURN_VOL_GRACE_MONTHS = 3
CHURN_VOL_MONTHLY_STEP = 0.1

TRANSITION_SHEET = "Transition Matrix"
TRANSITION_INTERVAL_LABEL = "Trading Days Between Transitions"
TRANSITION_PER_STD_LABEL = "Probability per Standard Deviation"
TRANSITION_Z_CAP_LABEL = "Standard Deviation Cap"
# Rows and columns are Bull, Churn, Bear: State 1, State 2, and State 3.
DEFAULT_TRANSITION = np.array([
    [0.90, 0.06, 0.04],
    [0.15, 0.70, 0.15],
    [0.20, 0.10, 0.70],
])
DEFAULT_TRANSITION_INTERVAL = TRADING_DAYS_PER_MONTH
# SMA Enter length used when the trigger asset does not name one. The sheet uses 210.
SMA_TREND_DAYS = 210
# Probability moved for each standard deviation from the SMA Enter average.
SMA_PROBABILITY_PER_STD = 0.10
SMA_Z_CAP = 3.0
DEFAULT_TRANSITION_NOTES = (
    "Highly likely to stay in Bull",
    "Moderate persistence, can break either way",
    "Sticky, but recovers rapidly directly to Bull",
)
_REGIME_ORDER = ("bull", "churn", "bear")
_REGIME_ALIASES = {
    "bull": "bull",
    "expansion": "bull",
    "state 1": "bull",
    "churn": "churn",
    "sideways": "churn",
    "state 2": "churn",
    "bear": "bear",
    "liquidation": "bear",
    "state 3": "bear",
}

EXPANSION = 1
CHURN = 2
LIQUIDATION = 3

CASH_FLIGHT_TICKERS = ("SPMO", "DWAS")
CASH_TICKER = "SGOV"

STATE_COLUMNS = (
    ("State 1 Drift", "State 1 Volatility"),
    ("State 2 Drift", "State 2 Volatility"),
    ("State 3 Drift", "State 3 Volatility"),
)

TICKER_ALIASES = {
    "BTC": "IBIT",
    "IBIT": "BTC",
    "VIIIX": "VOO",
}


def validate_transition_matrix(matrix):
    """A 3 by 3 matrix whose rows are Bull, Churn, and Bear and each sum to 1."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("transition matrix must be 3 by 3")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("transition probabilities must be finite and non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("each transition row must sum to 1")
    return matrix


def sma_zscore(prices, window):
    """Standard deviations of the last close from the average of the last `window` closes.

    `prices` is oldest first, one column per path. Above the average is positive
    and below it is negative. A short window, or a flat window, scores zero.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim == 1:
        prices = prices[:, None]
    window = int(window)
    if prices.shape[0] < window or window < 1:
        return np.zeros(prices.shape[1])
    sample = prices[-window:]
    sigma = sample.std(axis=0)
    score = np.zeros(prices.shape[1])
    np.divide(sample[-1] - sample.mean(axis=0), sigma, out=score, where=sigma > 0)
    return score


def tilt_transition(matrix, z, per_std=SMA_PROBABILITY_PER_STD, z_cap=SMA_Z_CAP):
    """Shift each row by the trigger's distance from its SMA Enter average.

    A positive score adds that much probability to Bull and removes it from
    Churn and Bear in proportion to the probability they already hold. A
    negative score takes it from Bull and gives it to Churn and Bear the same
    way. A row with neither Churn nor Bear splits a negative move equally, so
    both become more likely. The move stops at zero, z stops at ±z_cap, and
    each row still sums to one.
    """
    matrix = validate_transition_matrix(matrix)
    score = np.clip(np.atleast_1d(np.asarray(z, dtype=float)).astype(float).reshape(-1), -float(z_cap), float(z_cap))
    tilted = np.broadcast_to(matrix, (score.shape[0], 3, 3)).copy()
    bull = tilted[:, :, 0]
    churn = tilted[:, :, 1]
    bear = tilted[:, :, 2]
    other = churn + bear
    move = np.minimum(other, np.maximum((float(per_std) * score)[:, None], -bull))
    churn_share = np.full(churn.shape, 0.5)
    bear_share = np.full(bear.shape, 0.5)
    has_other = other > 0
    np.divide(churn, other, out=churn_share, where=has_other)
    np.divide(bear, other, out=bear_share, where=has_other)
    tilted[:, :, 0] = bull + move
    tilted[:, :, 1] = churn - move * churn_share
    tilted[:, :, 2] = bear - move * bear_share
    return tilted


def draw_next_states(states, matrix, rng=None):
    """Next regime for each path. `matrix[i, j]` is the probability of state j+1 after state i+1.

    `matrix` is one 3 by 3 block, or one block per path when each path has its own row probabilities.
    """
    states = np.asarray(states, dtype=int)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 2:
        if states.size and np.any((states < 1) | (states > matrix.shape[0])):
            raise ValueError("regime states must be 1, 2, or 3")
        per_path = np.broadcast_to(matrix, (states.shape[0],) + matrix.shape)
    elif matrix.ndim == 3 and matrix.shape[1:] == (3, 3) and matrix.shape[0] == states.shape[0]:
        if states.size and np.any((states < 1) | (states > 3)):
            raise ValueError("regime states must be 1, 2, or 3")
        per_path = matrix
    else:
        raise ValueError("transition matrix must be 3 by 3, or one 3 by 3 block per path")
    if states.shape[0] == 0:
        return np.zeros(0, dtype=int)
    cdf = np.cumsum(per_path, axis=2)
    cdf[:, :, -1] = 1.0
    draws = _uniforms(rng, states.shape[0])
    rows = cdf[np.arange(states.shape[0]), states - 1]
    chosen = np.full(states.shape, per_path.shape[-1], dtype=int)
    remaining = np.ones(states.shape, dtype=bool)
    last = per_path.shape[-1] - 1
    for column in range(last):
        take = remaining & (draws < rows[:, column])
        chosen[take] = column + 1
        remaining[take] = False
    chosen[remaining] = last + 1
    return chosen


def _uniforms(rng, count):
    if count == 0:
        return np.zeros(0)
    if rng is None:
        return np.random.random(count)
    return np.asarray(rng.random(count), dtype=float).reshape(count)


# A trailing six-month market return at or above this is a bull month.
STATE_BULL_TRAIL = 0.08
# A trailing six-month market return at or below this is a bear month.
STATE_BEAR_TRAIL = -0.10
STATE_TRAIL_MONTHS = 6


def estimate_state_parameters(asset, market, trail_months=STATE_TRAIL_MONTHS, bull_trail=STATE_BULL_TRAIL, bear_trail=STATE_BEAR_TRAIL):
    """Annual drift and volatility of `asset` in bull, churn, and bear markets.

    The market's trailing return, ending in the month being measured, labels
    that month. A rising month whose trail is at or above `bull_trail` is
    State 1. A falling month whose trail is at or below `bear_trail` is State
    3. The months between those trails, including rebounds, are State 2.
    Drift is the annualized mean of simple monthly returns, the mu in
    dS/S = mu dt + sigma dW. Volatility is the annualized standard deviation
    of monthly log returns.
    """
    asset = pd.Series(asset, dtype=float).resample("ME").last().dropna()
    market = pd.Series(market, dtype=float).resample("ME").last().dropna()
    paired = pd.concat({"asset": asset, "market": market}, axis=1, sort=True).dropna()
    trail = paired["market"] / paired["market"].shift(int(trail_months)) - 1.0
    sample = pd.DataFrame({
        "trail": trail,
        "market": paired["market"].pct_change(),
        "simple": paired["asset"].pct_change(),
        "log": np.log(paired["asset"]).diff(),
    }).dropna()
    bull = (sample["trail"] >= float(bull_trail)) & (sample["market"] > 0)
    bear = (sample["trail"] <= float(bear_trail)) & (sample["market"] < 0)
    states = (
        ("State 1 Drift", "State 1 Volatility", bull),
        ("State 2 Drift", "State 2 Volatility", ~bull & ~bear),
        ("State 3 Drift", "State 3 Volatility", bear),
    )
    parameters = {}
    for drift_name, volatility_name, mask in states:
        rows = sample.loc[mask]
        if len(rows) < 2:
            raise ValueError(f"{drift_name} needs at least 2 months")
        parameters[drift_name] = float(rows["simple"].mean() * 12.0)
        deviation = float(rows["log"].std(ddof=1))
        parameters[volatility_name] = 0.0 if not np.isfinite(deviation) else float(deviation * np.sqrt(12.0))
    return parameters


def scale_state_parameters(asset_parameters, market_parameters, baseline):
    """Express `asset_parameters` in the scenario units of `baseline`.

    `baseline` is the market's drift and volatility on the behavior sheet.
    Each cell is that scenario value times the asset's historical parameter
    divided by the market's historical parameter for the same state. An asset
    that matched the market in every state keeps the scenario row.
    """
    scaled = {}
    for drift_name, volatility_name in STATE_COLUMNS:
        market_drift = float(market_parameters[drift_name])
        market_volatility = float(market_parameters[volatility_name])
        if market_drift == 0 or market_volatility == 0:
            raise ValueError(f"{drift_name} cannot scale from a zero market parameter")
        scaled[drift_name] = float(baseline[drift_name]) * float(asset_parameters[drift_name]) / market_drift
        scaled[volatility_name] = float(baseline[volatility_name]) * float(asset_parameters[volatility_name]) / market_volatility
    return scaled


def align_regime_parameters(frame, labels, cash_flight=CASH_FLIGHT_TICKERS, cash_ticker=CASH_TICKER):
    """Drift and volatility arrays shaped (state, asset), taken from the behavior rows.

    A bitcoin row is used as written. Cash flight replaces liquidation only
    for tickers named in `cash_flight`. Pass an empty sequence to keep every
    liquidation cell on the sheet.
    """
    lookup = {str(row["Ticker"]).strip(): row for _, row in frame.iterrows()}
    drift = np.zeros((3, len(labels)))
    volatility = np.zeros((3, len(labels)))
    for asset, label in enumerate(labels):
        row = _row_for_label(lookup, label)
        for state, (drift_column, vol_column) in enumerate(STATE_COLUMNS):
            drift[state, asset] = float(row[drift_column])
            volatility[state, asset] = float(row[vol_column])
    if cash_flight:
        drift, volatility = apply_cash_flight(drift, volatility, labels, frame, cash_flight, cash_ticker)
    return drift, volatility


def apply_cash_flight(drift, volatility, labels, frame, cash_flight=CASH_FLIGHT_TICKERS, cash_ticker=CASH_TICKER):
    """In liquidation, momentum assets earn the cash ticker's return instead of their own."""
    lookup = {str(row["Ticker"]).strip(): row for _, row in frame.iterrows()}
    if cash_ticker not in lookup:
        return drift, volatility
    cash = lookup[cash_ticker]
    cash_drift = float(cash["State 3 Drift"])
    cash_vol = float(cash["State 3 Volatility"])
    flight = {_ticker_key(ticker) for ticker in cash_flight}
    drift = np.array(drift, dtype=float, copy=True)
    volatility = np.array(volatility, dtype=float, copy=True)
    for asset, label in enumerate(labels):
        if _ticker_key(label) in flight:
            drift[LIQUIDATION - 1, asset] = cash_drift
            volatility[LIQUIDATION - 1, asset] = cash_vol
    return drift, volatility


def churn_volatility_addon(days_in_churn, grace_months=CHURN_VOL_GRACE_MONTHS, monthly_step=CHURN_VOL_MONTHLY_STEP):
    """Extra trigger volatility after `grace_months` of churn, then `monthly_step` each month."""
    completed_months = np.asarray(days_in_churn) // TRADING_DAYS_PER_MONTH
    months_over = np.maximum(completed_months - (int(grace_months) - 1), 0)
    return months_over * float(monthly_step)


class RegimeBook:
    """Hidden regime state. Trading asks for today's drift and volatility, then the matrix may switch the regime."""

    def __init__(self, drift, volatility, num_paths, trigger_index, initial_states=None, transition=None, transition_interval=DEFAULT_TRANSITION_INTERVAL, rng=None, sma_window=SMA_TREND_DAYS, sma_per_std=SMA_PROBABILITY_PER_STD, sma_z_cap=SMA_Z_CAP, churn_vol_grace=CHURN_VOL_GRACE_MONTHS, churn_vol_step=CHURN_VOL_MONTHLY_STEP):
        self.drift = np.asarray(drift, dtype=float)
        self.volatility = np.asarray(volatility, dtype=float)
        self.trigger_index = trigger_index
        self.transition = validate_transition_matrix(DEFAULT_TRANSITION if transition is None else transition).copy()
        self.transition_interval = _as_trading_days(transition_interval)
        self.rng = rng
        self.sma_window = _as_trading_days(sma_window)
        self.sma_per_std = float(sma_per_std)
        self.sma_z_cap = float(sma_z_cap)
        self.churn_vol_grace = int(churn_vol_grace)
        self.churn_vol_step = float(churn_vol_step)
        self.states = np.full(num_paths, CHURN, dtype=int) if initial_states is None else np.asarray(initial_states, dtype=int).copy()
        self.days_in_churn = np.zeros(num_paths, dtype=int)
        self._step = 0
        self._trigger_price = np.ones(self.states.shape[0])
        self._trigger_samples = np.ones((self.sma_window, self.states.shape[0]))
        self._trigger_filled = 0

    def parameters(self):
        drift = self.drift[self.states - 1]
        volatility = self.volatility[self.states - 1].copy()
        in_churn = self.states == CHURN
        volatility[in_churn, self.trigger_index] += churn_volatility_addon(
            self.days_in_churn, self.churn_vol_grace, self.churn_vol_step
        )[in_churn]
        return drift, volatility

    def observe(self, trigger_growth=None):
        """Hold the current regime, then draw the next one when the interval elapses.

        The trigger price is the asset's market growth, compounded from 1. Once
        that price covers a full SMA Enter window, its standard deviations from
        the window average tilt the row: above the average raises Bull, and
        below the average raises Churn and Bear.
        """
        previous = self.states.copy()
        self._step += 1
        if trigger_growth is not None:
            growth = np.asarray(trigger_growth, dtype=float).reshape(self._trigger_price.shape)
            self._trigger_price = self._trigger_price * growth
            if self._trigger_filled < self.sma_window:
                self._trigger_samples[self._trigger_filled] = self._trigger_price
                self._trigger_filled += 1
            else:
                self._trigger_samples[:-1] = self._trigger_samples[1:]
                self._trigger_samples[-1] = self._trigger_price
        if self._step % self.transition_interval == 0:
            matrix = self.transition
            if self._trigger_filled >= self.sma_window:
                score = sma_zscore(self._trigger_samples, self.sma_window)
                matrix = tilt_transition(self.transition, score, self.sma_per_std, self.sma_z_cap)
            self.states = draw_next_states(self.states, matrix, self.rng)
        self.days_in_churn = np.where(previous == CHURN, self.days_in_churn + 1, 0)
        self.days_in_churn = np.where(self.states == CHURN, self.days_in_churn, 0)


def load_regime_parameters(path, labels, sheet="Asset State Behaviors"):
    frame = pd.read_excel(path, sheet_name=sheet)
    return align_regime_parameters(frame, labels)


def transition_sheet_rows(path, sheet=TRANSITION_SHEET):
    """Every cell on the transition sheet, formulas left as text."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        if sheet not in workbook.sheetnames:
            raise ValueError(f"{path} has no {sheet} sheet")
        return [tuple(row) for row in workbook[sheet].iter_rows(values_only=True)]
    finally:
        workbook.close()


def read_labeled_cell(rows, label):
    """Whether `label` is on the sheet, and the cell to its right.

    A missing label is `(False, None)`. A label with an empty neighbor is
    `(True, None)`, which is a value the sheet set on purpose.
    """
    key = label.strip().lower()
    for row in rows:
        for index, cell in enumerate(row):
            if cell is None or str(cell).strip().lower() != key:
                continue
            if index + 1 >= len(row):
                return True, None
            return True, row[index + 1]
    return False, None


def load_transition_matrix(path, sheet=TRANSITION_SHEET):
    """Bull, Churn, and Bear rows from the transition sheet, plus the trading-day interval."""
    rows = transition_sheet_rows(path, sheet)
    return _matrix_from_rows(rows), _interval_from_rows(rows), _shift_per_std(rows), _z_cap(rows)


def add_transition_sheet(workbook, sheet_name=TRANSITION_SHEET):
    """Write the default Bull / Churn / Bear matrix. Replaces an existing sheet of the same name."""
    if sheet_name in workbook.sheetnames:
        index = workbook.sheetnames.index(sheet_name)
        del workbook[sheet_name]
    elif "Asset State Behaviors" in workbook.sheetnames:
        index = workbook.sheetnames.index("Asset State Behaviors") + 1
    else:
        index = len(workbook.sheetnames)
    sheet = workbook.create_sheet(sheet_name, index)
    headers = ["From", "Bull", "Churn", "Bear", "Note", "Expected Duration (transitions)"]
    for column, header in enumerate(headers, start=1):
        sheet.cell(1, column, header)
    for row, (name, probabilities, note) in enumerate(
        zip(("Bull", "Churn", "Bear"), DEFAULT_TRANSITION, DEFAULT_TRANSITION_NOTES),
        start=2,
    ):
        sheet.cell(row, 1, name)
        for column, probability in enumerate(probabilities, start=2):
            cell = sheet.cell(row, column, float(probability))
            cell.number_format = "0%"
        sheet.cell(row, 5, note)
        duration = sheet.cell(row, 6, f"=1/(1-{openpyxl.utils.get_column_letter(row)}{row})")
        duration.number_format = "0.0"
    sheet["A6"] = TRANSITION_INTERVAL_LABEL
    sheet["B6"] = DEFAULT_TRANSITION_INTERVAL
    sheet["B6"].number_format = "0"
    sheet["A7"] = "Each path draws its next regime from the current row once per this many trading days. 21 is one month."
    sheet["A8"] = "Bull is State 1 (Expansion), Churn is State 2, and Bear is State 3 (Liquidation)."
    sheet["A10"] = TRANSITION_PER_STD_LABEL
    sheet["B10"] = SMA_PROBABILITY_PER_STD
    sheet["B10"].number_format = "0%"
    sheet["A11"] = TRANSITION_Z_CAP_LABEL
    sheet["B11"] = SMA_Z_CAP
    sheet["B11"].number_format = "0.0"
    sheet["A12"] = (
        "The trigger close is scored against its SMA Enter average. "
        "Each standard deviation moves this much probability toward Bull when the close is above that average, "
        "and toward Churn and Bear when it is below. Churn and Bear share the move in proportion to the probability they already have. "
        "The move stops at the cap and at zero."
    )
    _write_tilted_block(sheet, 14, 1)
    _write_tilted_block(sheet, 20, -1)
    sheet.column_dimensions["A"].width = 42
    sheet.column_dimensions["E"].width = 56
    sheet.column_dimensions["F"].width = 34
    table = Table(displayName="TransitionMatrix", ref="A1:E4")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    sheet.add_table(table)
    return sheet


def _write_tilted_block(sheet, start_row, z_score):
    """Formulas for the base matrix after a fixed SMA Enter z-score."""
    title = f"Transition at {'+' if z_score > 0 else ''}{z_score:g} Standard Deviation"
    sheet.cell(start_row, 1, title)
    headers = ["From", "Bull", "Churn", "Bear"]
    for column, header in enumerate(headers, start=1):
        sheet.cell(start_row + 1, column, header)
    for offset, name in enumerate(("Bull", "Churn", "Bear")):
        source = 2 + offset
        row = start_row + 2 + offset
        sheet.cell(row, 1, name)
        for column, letter in enumerate(("B", "C", "D"), start=2):
            cell = sheet.cell(row, column, _tilted_formula(source, letter, z_score))
            cell.number_format = "0%"


def _tilted_formula(source_row, letter, z_score):
    shift = f"$B$10*MEDIAN(-$B$11,{z_score:g},$B$11)"
    move = f"MIN(C{source_row}+D{source_row},MAX({shift},-B{source_row}))"
    if letter == "B":
        return f"=B{source_row}+{move}"
    return (
        f"={letter}{source_row}-{move}"
        f"*IF(C{source_row}+D{source_row}=0,0.5,{letter}{source_row}/(C{source_row}+D{source_row}))"
    )


def _shift_per_std(rows):
    return _labeled_float(rows, TRANSITION_PER_STD_LABEL, SMA_PROBABILITY_PER_STD)


def _z_cap(rows):
    return _labeled_float(rows, TRANSITION_Z_CAP_LABEL, SMA_Z_CAP)


def _labeled_float(rows, label, default):
    key = label.strip().lower()
    for row in rows:
        for index, cell in enumerate(row):
            if cell is None or str(cell).strip().lower() != key:
                continue
            if index + 1 >= len(row) or row[index + 1] is None:
                raise ValueError(f"{label} needs a value in the next cell")
            value = float(row[index + 1])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be zero or positive")
            return value
    return default


def _matrix_from_rows(rows):
    header_index, columns, from_index = _transition_header(rows)
    matrix = np.zeros((3, 3))
    seen = set()
    for row in rows[header_index + 1:]:
        if from_index >= len(row):
            continue
        name = _canonical_regime(row[from_index])
        if name not in _REGIME_ORDER:
            continue
        if name in seen:
            if seen == set(_REGIME_ORDER):
                break
            raise ValueError(f"Transition Matrix lists {name} more than once")
        seen.add(name)
        origin = _REGIME_ORDER.index(name)
        for destination, column in columns.items():
            if column >= len(row) or row[column] is None:
                raise ValueError(f"Transition Matrix is missing the {destination} probability for {name}")
            matrix[origin, _REGIME_ORDER.index(destination)] = float(row[column])
    missing = [name for name in _REGIME_ORDER if name not in seen]
    if missing:
        raise ValueError(f"Transition Matrix is missing {', '.join(missing)}")
    return validate_transition_matrix(matrix)


def _transition_header(rows):
    for index, row in enumerate(rows):
        labels = [_canonical_regime(cell) for cell in row]
        if set(_REGIME_ORDER) <= set(labels):
            columns = {}
            for column, label in enumerate(labels):
                if label not in _REGIME_ORDER:
                    continue
                if label in columns:
                    raise ValueError(f"Transition Matrix has more than one {label} column")
                columns[label] = column
            from_index = 0
            for column, cell in enumerate(row):
                if cell is not None and str(cell).strip().lower() == "from":
                    from_index = column
                    break
            return index, columns, from_index
    raise ValueError("Transition Matrix sheet needs Bull, Churn, and Bear columns")


def _interval_from_rows(rows):
    label = TRANSITION_INTERVAL_LABEL.strip().lower()
    for row in rows:
        for index, cell in enumerate(row):
            if cell is not None and str(cell).strip().lower() == label:
                if index + 1 >= len(row) or row[index + 1] is None:
                    raise ValueError(f"{TRANSITION_INTERVAL_LABEL} needs a value in the next cell")
                return _as_trading_days(row[index + 1])
    raise ValueError(f"Transition Matrix sheet needs a {TRANSITION_INTERVAL_LABEL} value")


def _as_trading_days(value):
    number = float(value)
    if not number.is_integer() or number < 1:
        raise ValueError("transition interval must be a positive whole number of trading days")
    return int(number)


def _canonical_regime(value):
    if value is None:
        return None
    return _REGIME_ALIASES.get(str(value).strip().lower())


def trigger_index(labels, ticker="VOO"):
    for index, label in enumerate(labels):
        if _ticker_key(label) == ticker:
            return index
    raise KeyError(f"{ticker} is not among {list(labels)}")


def _ticker_key(label):
    text = str(label).strip()
    if text in TICKER_ALIASES:
        return TICKER_ALIASES[text]
    for part in text.split("/"):
        part = part.strip()
        if part in ("VOO", "SPMO", "DWAS", "TQQQ", "CSVYX", "DBMF", "BTC", "IBIT", "AVES", "KMLM", "GLD", "SGOV"):
            return TICKER_ALIASES.get(part, part)
    return text


# The behavior sheet still names the equity sleeve SPMO. VOO/VIIX uses that row.
BEHAVIOR_ALIASES = {
    "VOO": "SPMO",
}


def _row_for_label(lookup, label):
    text = str(label).strip()
    if text in lookup:
        return lookup[text]
    key = _ticker_key(label)
    row = _row_for_key(lookup, key)
    if row is not None:
        return row
    alias = BEHAVIOR_ALIASES.get(key)
    if alias is not None:
        row = _row_for_key(lookup, alias)
        if row is not None:
            return row
    raise KeyError(f"{label} has no row on Asset State Behaviors")


def _row_for_key(lookup, key):
    for ticker, row in lookup.items():
        if str(ticker).strip() == key or _ticker_key(ticker) == key:
            return row
    return None
