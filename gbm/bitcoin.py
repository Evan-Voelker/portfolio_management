import numpy as np

# Next-decade regimes. The median path under the workbook's monthly transition
# (bull stays with probability 0.88: about 59% expansion, 21% churn, 20%
# liquidation) compounds near 26%. That is Fidelity's 2026 planning return
# of 25%, and it sits just under Bitwise's 28.3% ten-year CAGR and a
# power-law decade of about 29% (exponent 5.69, age 17.8 years in October
# 2026). VanEck's 15% is the 25-year base case, so it describes the years
# after this decade.
#
# Expansion and churn both keep an adoption premium. Bitwise treats the
# four-year cycle as over, so a sideways equity market does not zero it out.
# A liquidation year has a median near -44%: Timmer's matured winter of
# about 50%, and the drawdown BlackRock described from the October 2025 high,
# rather than an 80% launch-era winter.
#
# Volatility averages about 42% across those regimes. That is above Bitwise's
# 32.9% decade-midpoint pin and below Fidelity's 50% sizing vol. The crash
# state, 60%, is inside VanEck's 40-70% long-term range.
MATURE_BTC_DRIFT = np.array([0.54, 0.43, -0.40])
MATURE_BTC_VOLATILITY = np.array([0.35, 0.45, 0.60])
# In a liquidation, correlation with each risky asset moves from BTC's own
# history toward this level. A 0.20 history lands near 0.35, under the 0.50
# that Bitwise says a 90-day equity correlation has rarely exceeded.
# Cash is left on its historical correlation. Bitwise's bond forecast is about 0.
MATURE_BTC_STRESS_CORRELATION = 0.45
MATURE_BTC_STRESS_BLEND = 0.60

_BITCOIN = {"BTC", "IBIT"}
_CASH = {"SGOV"}


def is_bitcoin(label):
    return _symbol(label) in _BITCOIN


def is_cash(label):
    return _symbol(label) in _CASH


def mature_return_center():
    """12-month simple return of the expansion drift. A year at that drift has no momentum gap."""
    return float(np.exp(MATURE_BTC_DRIFT[0]) - 1.0)


def stationary_weights(transition):
    """Share of time spent in each regime once the monthly chain has settled."""
    matrix = np.asarray(transition, dtype=float)
    width = matrix.shape[0]
    system = np.vstack([matrix.T - np.eye(width), np.ones(width)])
    target = np.zeros(width + 1)
    target[-1] = 1.0
    weights, *_ = np.linalg.lstsq(system, target, rcond=None)
    return weights


def median_log_growth(weights):
    """Log growth of the median path when each regime gets `weights` of the decade.

    Geometric growth in a regime is its drift minus half its variance.
    """
    weights = np.asarray(weights, dtype=float)
    geometric = MATURE_BTC_DRIFT - 0.5 * np.square(MATURE_BTC_VOLATILITY)
    return float(weights @ geometric)


def apply_mature_bitcoin(drift, volatility, labels):
    """Write the published mature-asset regimes over bitcoin columns.

    The simulation does not call this. Asset State Behaviors is the drift
    the run uses. This remains the vector those cells were aligned to.
    """
    drift = np.array(drift, dtype=float, copy=True)
    volatility = np.array(volatility, dtype=float, copy=True)
    for index, label in enumerate(labels):
        if not is_bitcoin(label):
            continue
        drift[:, index] = MATURE_BTC_DRIFT
        volatility[:, index] = MATURE_BTC_VOLATILITY
    return drift, volatility


def restress_bitcoin(baseline, liquidation, labels, stress_correlation=None, stress_blend=None):
    """Move BTC toward risky assets in a crash, and keep its correlation with cash.

    The flat liquidation blend pulls every pair, including BTC and SGOV, toward 0.85.
    A maturing BTC joins the risk complex partway and still diversifies against cash.
    `stress_correlation` and `stress_blend` replace the published pair when the
    transition sheet names them.
    """
    labels = list(labels)
    bitcoin = [index for index, label in enumerate(labels) if is_bitcoin(label)]
    if not bitcoin:
        return liquidation
    correlation = MATURE_BTC_STRESS_CORRELATION if stress_correlation is None else float(stress_correlation)
    blend = MATURE_BTC_STRESS_BLEND if stress_blend is None else float(stress_blend)
    updated = np.array(liquidation, dtype=float, copy=True)
    baseline = np.asarray(baseline, dtype=float)
    width = updated.shape[0]
    for btc in bitcoin:
        for other in range(width):
            if other == btc:
                continue
            historical = baseline[btc, other]
            if is_cash(labels[other]):
                value = historical
            else:
                value = (1.0 - blend) * historical + blend * correlation
            updated[btc, other] = value
            updated[other, btc] = value
    np.fill_diagonal(updated, 1.0)
    return updated


def _symbol(label):
    text = str(label).strip()
    parts = [text]
    parts.extend(part.strip() for part in text.split("/") if part.strip())
    for part in parts:
        if part in _BITCOIN or part in _CASH:
            return part
    return text
