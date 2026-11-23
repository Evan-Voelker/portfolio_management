import numpy as np

from gbm.asset_simulation_matrix import market_ticker

# Live monthly correlation of DBMF with VOO, May 2019 through October 2026.
# Ordinary months sit near -0.10. A deep equity decline sits near -0.40,
# between the -0.22 of every down month and the -0.47 of months worse than -5%.
# Churn damps that ordinary correlation the same way it damps every other pair.
EXPANSION_EQUITY_CORRELATION = -0.10
CHURN_EQUITY_CORRELATION = -0.075
LIQUIDATION_EQUITY_CORRELATION = -0.40
# An asset this correlated with VOO is equity beta. Managed futures should
# not follow it into a crash. Gold, bitcoin, and cash sit well below this.
EQUITY_BETA_THRESHOLD = 0.50

_MANAGED_FUTURES = {"DBMF", "KMLM"}
_CASH = {"SGOV"}
_EQUITY = {
    "VOO", "VIIX", "VIIIX", "SPMO", "DWAS", "CSVYX", "AVES", "TQQQ", "VXUS",
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU",
    "QQQ", "SPY", "VTI",
}


def is_dbmf(label):
    return market_ticker(label) == "DBMF"


def restress_managed_futures(baseline, expansion, churn, liquidation, labels, equity=None):
    """Give DBMF the live fund's equity correlation, and keep its other pairs.

    The flat liquidation blend pulls every pair toward 0.85. A managed-futures
    fund is the holding that should do the opposite: near zero or slightly
    negative with equities in a normal month, and more negative in a crash.
    Cash, gold, bitcoin, and the other CTA fund stay on their own history.
    A book with no DBMF is returned unchanged. `equity` is the expansion,
    churn, and liquidation correlation with equities. The transition sheet
    replaces the published triple when it names one.
    """
    labels = list(labels)
    managed = [index for index, label in enumerate(labels) if is_dbmf(label)]
    if not managed:
        return expansion, churn, liquidation

    baseline = np.asarray(baseline, dtype=float)
    voo = next((index for index, label in enumerate(labels) if market_ticker(label) == "VOO"), None)
    equity_names = [
        index for index, label in enumerate(labels)
        if _is_equity(label, baseline, voo, index)
    ]
    targets = (
        EXPANSION_EQUITY_CORRELATION,
        CHURN_EQUITY_CORRELATION,
        LIQUIDATION_EQUITY_CORRELATION,
    ) if equity is None else tuple(float(value) for value in equity)
    updated = [np.array(matrix, dtype=float, copy=True) for matrix in (expansion, churn, liquidation)]
    for matrix, target, restore_history in zip(updated, targets, (False, False, True)):
        for asset in managed:
            for other in range(len(labels)):
                if other == asset:
                    continue
                if other in equity_names:
                    value = target
                elif restore_history:
                    value = baseline[asset, other]
                else:
                    continue
                matrix[asset, other] = value
                matrix[other, asset] = value
        np.fill_diagonal(matrix, 1.0)
    return tuple(updated)


def _is_equity(label, baseline, voo, index):
    symbol = market_ticker(label)
    if symbol in _MANAGED_FUTURES or symbol in _CASH:
        return False
    if symbol in _EQUITY:
        return True
    if voo is None:
        return False
    return float(baseline[voo, index]) >= EQUITY_BETA_THRESHOLD
