import numpy as np
import pandas as pd
from gbm.gbm_step import load_simulation_assumptions
from gbm.regime import align_regime_parameters, load_transition_matrix
from gbm.expenses import apply_expense_ratios, load_expense_ratios, overlay_model_expenses

wb = "portfolio_gbm_calculator.xlsx"
labels = ["VOO", "VXUS", "SGOV", "DBMF", "BTC"]
behavior = pd.read_excel(wb, sheet_name="Asset State Behaviors")
print(behavior[behavior["Ticker"].isin(labels)].to_string(index=False))
model = pd.read_excel(wb, sheet_name="Asset Parameters")
cols = [c for c in ["Ticker", "Target Allocation", "Lower Band", "Upper Band", "SMA Exit", "SMA-exclusion", "SMA Enter", "Asset Type"] if c in model.columns]
print(model[cols].to_string(index=False))
assumptions = load_simulation_assumptions(wb)
print(
    "liq corr", assumptions.liquidation_correlation,
    "grace", assumptions.churn_vol_grace_months,
    "step", assumptions.churn_vol_monthly_step,
    "btc", assumptions.bitcoin_stress_correlation, assumptions.bitcoin_stress_blend,
    "dbmf liq", assumptions.dbmf_liquidation_equity,
)
trans, interval, per_std, zcap = load_transition_matrix(wb)
print(np.array2string(trans, precision=4, floatmode="fixed"))
print("interval", interval, "per_std", per_std, "zcap", zcap)
weights, *_ = np.linalg.lstsq(np.vstack([trans.T - np.eye(3), np.ones(3)]), np.array([0., 0, 0, 1]), rcond=None)
print("stationary", np.round(weights, 4))
names = ["Expansion", "Churn", "Liquidation"]
for i, name in enumerate(names):
    stay = float(trans[i, i])
    spell_mo = 1 / (1 - stay)
    print(f"{name} spell {spell_mo:.2f} mo  P(>=12) {stay ** 12:.1%}  share {weights[i]:.1%}")
print(f"years between bears {((1 / (1 - trans[2, 2])) / weights[2]) / 12:.2f}")
fees = overlay_model_expenses(load_expense_ratios(wb), wb, "Asset Parameters")
drift, vol = align_regime_parameters(behavior, labels, cash_flight=())
drift = np.vstack([apply_expense_ratios(drift[s], labels, fees) for s in range(3)])
log_g = drift - 0.5 * vol ** 2
simple = np.exp(log_g) - 1
print("\nMEDIANS")
for i, lab in enumerate(labels):
    print(lab)
    for s, name in enumerate(names):
        spell = 1 / (1 - float(trans[s, s]))
        spell_ret = float(np.exp(log_g[s, i] * spell / 12) - 1)
        print(f"  {name:12} drift {drift[s, i]:7.3f}  vol {vol[s, i]:.0%}  annual {simple[s, i]:7.2%}  spell {spell_ret:7.2%}")
    print(f"  cycle {float(np.exp(weights @ log_g[:, i]) - 1):7.2%}")
outside = float(trans[0, 1] + trans[0, 2])
print(f"bull exit {outside:.2%} max tilt {per_std * zcap:.2%} seals {outside <= per_std * zcap + 1e-9}")
mix = np.array([0.6, 0, 0, 0.2, 0.2])
print("60/20/20")
for s, name in enumerate(names):
    print(f"  {name} {float(mix @ simple[s]):7.2%}")
print(f"  cycle {float(np.exp(weights @ np.log1p(simple @ mix)) - 1):7.2%}")
